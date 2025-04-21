import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig
from copy import deepcopy
from PIL import Image
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt


#################################################################################
#                             Dataset Part                          #
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
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device = torch.device("cuda")

    # Load SD3.5 Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    vae = pipe.vae
    unet = pipe.unet
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Insert LoRA into UNet
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["Transformer2DModel", "Attention"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="UNET"
    )
    unet = get_peft_model(unet, lora_config)

    # Freeze VAE
    vae.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # Setup data:
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

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                input_latent = vae.encode(images.repeat(1,3,1,1)).latent_dist.sample().mul(0.18215)
                target_latent = vae.encode(targets.repeat(1,3,1,1)).latent_dist.sample().mul(0.18215)

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

        # Save checkpoint every epoch
        os.makedirs(args.output_dir, exist_ok=True)
        unet.save_pretrained(os.path.join(args.output_dir, f"lora_epoch_{epoch}.safetensors"))
        
        # Inference and save visualization every epoch
        unet.eval()
        with torch.no_grad():
            num_samples = 10
            visual_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
            images, targets = next(iter(visual_loader))
            images, targets = images.to(device), targets.to(device)

            input_latent = vae.encode(images.repeat(1,3,1,1)).latent_dist.sample().mul(0.18215)

            noise = torch.randn_like(input_latent)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (input_latent.size(0),), device=device)
            noisy_latents = scheduler.add_noise(input_latent, noise, timesteps)

            generated_latents = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=input_latent
            ).sample

            generated_images = vae.decode(generated_latents / 0.18215).sample
            generated_images = 0.299 * generated_images[:, 0:1, :, :] + 0.587 * generated_images[:, 1:2, :, :] + 0.114 * generated_images[:, 2:3, :, :]

            # Save visualization
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
            for i in range(num_samples):
                axes[0][i].imshow(images[i].cpu().squeeze(), cmap="gray")
                axes[0][i].axis("off")
                axes[1][i].imshow(generated_images[i].cpu().squeeze(), cmap="gray")
                axes[1][i].axis("off")
                axes[2][i].imshow(targets[i].cpu().squeeze(), cmap="gray")
                axes[2][i].axis("off")
            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.suptitle("Epoch {}: Input | Generated | Target".format(epoch))
            plt.savefig(os.path.join(args.output_dir, f"sample_epoch{epoch}.png"))
            plt.close()

        unet.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to pre-trained SD3.5 model.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", type=str, default="contents", help="Where to save lora adapters.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    main(args)
