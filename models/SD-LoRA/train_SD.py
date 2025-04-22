import os
import math
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda")


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


def get_texture_folders(root_dir):
    return [os.path.join(root_dir, texture) for texture in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, texture))]


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


def main(args):
    weight_dtype = torch.bfloat16
    snr_gamma = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    paired_transform = PairedRandomFlip(p=0.5)
    dataset = TextureDataset(args.data_dir, transform=transform, paired_transform=paired_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Starting training for {args.epochs} epochs...")

    model_path = args.model_path
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=model_path, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=model_path, subfolder="unet", torch_dtype=weight_dtype)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="UNET"
    )
    unet = get_peft_model(unet, lora_config)

    vae.to(DEVICE, dtype=weight_dtype)
    unet.to(DEVICE, dtype=weight_dtype)

    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = get_scheduler(
        name="cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=2000,
        num_cycles=3
    )

    global_step = 0
    max_train_steps = 2000
    progress_bar = tqdm(range(max_train_steps), desc="Training Steps")

    for epoch in range(math.ceil(max_train_steps / len(loader))):
        unet.train()

        for input_images, target_images in loader:
            if global_step >= max_train_steps:
                break

            input_images = input_images.to(DEVICE, dtype=weight_dtype)
            target_images = target_images.to(DEVICE, dtype=weight_dtype)

            with torch.no_grad():
                input_latents = vae.encode(input_images.repeat(1, 3, 1, 1)).latent_dist.sample() * vae.config.scaling_factor
                target_latents = vae.encode(target_images.repeat(1, 3, 1, 1)).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (target_latents.shape[0],), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

            encoder_hidden_states = torch.zeros((target_latents.shape[0], 77, 768), device=DEVICE, dtype=weight_dtype)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if not snr_gamma:
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                mse_loss_weights = mse_loss_weights / snr
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

            if global_step % 500 == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                unet.save_pretrained(os.path.join(save_path, "unet"))

        # 推理与可视化
        unet.eval()
        with torch.no_grad():
            num_samples = 10
            visualization_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
            image, heightmap = next(iter(visualization_loader))
            image, heightmap = image.to(DEVICE), heightmap.to(DEVICE)

            z = vae.encode(image.repeat(1, 3, 1, 1)).latent_dist.sample() * vae.config.scaling_factor
            dummy_cond = torch.zeros((num_samples, 77, 768), device=DEVICE, dtype=weight_dtype)

            timesteps_vis = torch.randint(0, noise_scheduler.config.num_train_timesteps, (num_samples,), device=DEVICE).long()
            noise_vis = torch.randn_like(z)
            noisy_z = noise_scheduler.add_noise(z, noise_vis, timesteps_vis)
            samples = unet(noisy_z, timesteps_vis, dummy_cond).sample
            samples = vae.decode(samples / vae.config.scaling_factor).sample
            samples = 0.299 * samples[:, 0:1] + 0.587 * samples[:, 1:2] + 0.114 * samples[:, 2:3]

            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
            for i in range(num_samples):
                axes[0][i].imshow(image[i].squeeze().cpu().numpy(), cmap="gray")
                axes[0][i].axis("off")
                axes[1][i].imshow(samples[i].squeeze().cpu().numpy(), cmap="gray")
                axes[1][i].axis("off")
                axes[2][i].imshow(heightmap[i].squeeze().cpu().numpy(), cmap="gray")
                axes[2][i].axis("off")

            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.suptitle(f"Height maps generated by UNet + LoRA (Epoch {epoch})")
            plt.savefig(os.path.join(args.output_dir, f"sample_epoch{epoch}.png"))
            plt.close()

    save_path = os.path.join(args.output_dir, "checkpoint-last")
    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(os.path.join(save_path, "unet"))
    print("Training complete!\\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="pretrained_models", help="Path to store/load pretrained SD2.1 components")
    parser.add_argument("--output-dir", type=str, default="lora_outputs")
    parser.add_argument("--data-dir", type=str, default="../../Texture")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
