import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DiffusionPipeline, DDPMScheduler
from lora_diffusion import inject_trainable_lora, extract_lora_ups_down
from copy import deepcopy
from PIL import Image
from huggingface_hub import hf_hub_download, whoami
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

#################################################################################
#                             Dataset Part                                     #
#################################################################################

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
    device = torch.device("cuda")

    # Load SDXL Pipeline
    try:
        info = whoami()
        print(f"Authenticated to Huggingface Hub as: {info['name']}")
    except Exception as e:
        raise RuntimeError("You are not logged into Huggingface Hub. Please run 'huggingface-cli login' first.")

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                             cache_dir=args.model_path, 
                                             torch_dtype=torch.float16, variant="fp16").to(device)

    vae = pipe.vae
    unet = pipe.unet
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    if hasattr(pipe, 'text_encoder'):
        pipe.text_encoder.requires_grad_(False)

    # Inject LoRA into UNet
    lora_params, train_names = inject_trainable_lora(unet, r=args.lora_rank, lora_alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Optimizer
    optimizer = torch.optim.AdamW(lora_params.parameters(), lr=args.lr)

    # Setup dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    paired_transform = PairedRandomFlip(p=0.5)
    root_dir = "../../Texture"
    dataset = TextureDataset(root_dir, transform=transform, paired_transform=paired_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Starting training for {args.epochs} epochs...")

    global_step = 0
    for epoch in range(args.epochs):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                input_latent = vae.encode(images).latent_dist.sample().mul(0.18215)
                target_latent = vae.encode(targets).latent_dist.sample().mul(0.18215)

            noise = torch.randn_like(target_latent)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (target_latent.size(0),), device=device)
            noisy_latents = scheduler.add_noise(target_latent, noise, timesteps)

            preds = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=input_latent
            ).sample

            loss = F.mse_loss(preds, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.log_every == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")

            global_step += 1

        # Save LoRA adapters every epoch
        os.makedirs(args.output_dir, exist_ok=True)
        extract_lora_ups_down(unet, save_directory=args.output_dir, save_name=f"lora_epoch_{epoch}.safetensors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="pretrained_models", help="Path to pre-trained SDXL model cache.")
    parser.add_argument("--output-dir", type=str, default="contents", help="Where to save LoRA adapters.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    main(args)
