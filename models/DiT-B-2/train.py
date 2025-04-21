# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import random

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_texture_folders(root_dir):
    return [os.path.join(root_dir, texture) for texture in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, texture))]


class PairedRandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            target = transforms.functional.hflip(target)
        if random.random() < self.p:
            image = transforms.functional.vflip(image)
            target = transforms.functional.vflip(target)
        return image, target


class TextureDataset(Dataset):
    def __init__(self, root_dir, transform=None, paired_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.paired_transform = paired_transform
        self.file_pairs = self._load_file_pairs()
    
    def _load_file_pairs(self):
        file_pairs = []
        texture_folders = get_texture_folders(self.root_dir)
        
        for texture_folder in texture_folders:
            texture_name = texture_folder.split("\\")[1]
            
            files = os.listdir(texture_folder)
            base_names = set(f.split(".")[0] for f in files)
            
            for base in base_names:
                image_path = os.path.join(texture_folder, f"{base}.jpg")
                heightmap_path = os.path.join(texture_folder, f"{base}.pkl")
                
                if os.path.exists(image_path) and os.path.exists(heightmap_path):
                    file_pairs.append((image_path, heightmap_path))
        
        return file_pairs
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        image_path, heightmap_path = self.file_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        with open(heightmap_path, 'rb') as f:
            heightmap = pickle.load(f).astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            heightmap = self.transform(heightmap)
            
            h_max = heightmap.max()
            h_min = heightmap.min()
            heightmap = (255 * (heightmap - h_min) / (h_max - h_min)).clamp(0, 255).byte()
            heightmap = heightmap / 255
        
        if self.paired_transform:
            image, heightmap = self.paired_transform(image, heightmap)
        
        return image, heightmap


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    # experiment_index = len(glob(f"{args.results_dir}/*"))
    # model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    # # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    # checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    # os.makedirs(checkpoint_dir, exist_ok=True)

    # Create model:
    # "Image size must be divisible by 8 (for the VAE encoder)." (240, 320) // 8
    # padding for pachify
    latent_size = (30, 40)
    model = DiT_models[args.model](
        input_size=latent_size,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).cuda()  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.cuda()
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").cuda()
    vae_mse = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").cuda()
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    paired_transform = PairedRandomFlip(p=0.5)
    root_dir = "../../Texture"
    dataset = TextureDataset(root_dir, transform=transform, paired_transform=paired_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}...")
        for image, heightmap in loader:
            image = image.cuda()
            heightmap = heightmap.cuda()
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                image = vae.encode(image.repeat(1, 3, 1, 1)).latent_dist.sample().mul_(0.18215)
                heightmap = vae.encode(heightmap.repeat(1, 3, 1, 1)).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (heightmap.shape[0],)).cuda()
            loss_dict = diffusion.training_losses(model, heightmap, t, noise=image)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps)
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        model.eval()  # important! This disables randomized embedding dropout
        # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
        with torch.no_grad():
            num_samples = 10
            
            visualzation_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
            (image, heightmap) = next(iter(visualzation_loader))
            image, heightmap = image.cuda(), heightmap.cuda()
            z = vae.encode(image.repeat(1, 3, 1, 1)).latent_dist.sample().mul_(0.18215)

            samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, progress=True, device='cuda'
            )
            samples = vae_mse.decode(samples / 0.18215).sample
            samples = 0.299 * samples[:, 0:1, :, :] + 0.587 * samples[:, 1:2, :, :] + 0.114 * samples[:, 2:3, :, :]
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
            for i in range(num_samples):
                axes[0][i].imshow(image[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
                axes[0][i].axis("off")
                axes[1][i].imshow(samples[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
                axes[1][i].axis("off")
                axes[2][i].imshow(heightmap[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
                axes[2][i].axis("off")

            fig.tight_layout()
            plt.subplots_adjust(wspace = 0, hspace = 0)
            plt.suptitle("Height maps generated by DiT")
            plt.savefig(f"contents/sample_epoch{epoch}.png")
            
    torch.save(model.state_dict(), args.model + '.ckpt')



if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
