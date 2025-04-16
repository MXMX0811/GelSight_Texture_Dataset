# The code is implemented based on https://github.com/cloneofsimo/minRF
import pickle
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from thop import profile
from thop import clever_format
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from diffusers.models import AutoencoderKL

import wandb
from dit import DiT_Llama


class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, image, heightmap):
        x = heightmap
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        # z1 = torch.randn_like(x)
        z1 = image
        zt = (1 - texp) * x + texp * z1
        
        # flops, params = profile(self.model, inputs=(zt, t))
        # flops, params = clever_format([flops, params], "%.3f")
        # print("FLOPs: ", flops, ", parameters: ", params)
        
        vtheta = self.model(zt, t)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, sample_steps=50):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t)

            z = z - dt * vc
            images.append(z)
        # return images
        return z
    
    
def get_texture_folders(root_dir):
    return [os.path.join(root_dir, texture) for texture in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, texture))]


class TextureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
        
        return image, heightmap


if __name__ == "__main__":
    # train class conditional RF on mnist.
    channels = 4
    
    # FLOPs:  810.926G , parameters:  22.770M
    # model = DiT_Llama(
    #     channels, dim=256, n_layers=16, n_heads=32
    # ).cuda()
    
    # FLOPs:  73.973T , parameters:  4.598G
    # model = DiT_Llama(
    #     channels, dim=3072, n_layers=32, n_heads=32
    # ).cuda()
    
    # DiT_XL_2
    model = DiT_Llama(
        channels, dim=1152, n_layers=28, n_heads=16
    ).cuda()
    
    # model = DiT_Llama(
    #     channels, dim=512, n_layers=2, n_heads=4
    # ).cuda()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    
    root_dir = "../../Texture"
    dataset = TextureDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").cuda()
    vae_mse = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(rf.model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = torch.nn.MSELoss()
    
    num_epochs = 100

    wandb.init(project=f"rfDiT_texture")

    for epoch in range(num_epochs):
        rf.model.train()
        epoch_loss = 0
        for i, (image, heightmap) in tqdm(enumerate(dataloader)):
            image, heightmap = image.cuda(), heightmap.cuda()
            image = vae.encode(image.repeat(1, 3, 1, 1)).latent_dist.sample().mul_(0.18215)
            heightmap = vae.encode(heightmap.repeat(1, 3, 1, 1)).latent_dist.sample().mul_(0.18215)
            optimizer.zero_grad()
            loss, blsct = rf.forward(image, heightmap)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(rf.model.parameters(), max_norm=1.0)
            
            epoch_loss += loss.item()
            wandb.log({"loss": loss.item()})

        scheduler.step(epoch_loss/len(dataloader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

        wandb.log({f"Epoch Loss": epoch_loss/len(dataloader)})

        rf.model.eval()
        with torch.no_grad():
            num_samples = 10
            
            visualzation_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
            (image, heightmap) = next(iter(visualzation_loader))
            image, heightmap = image.cuda(), heightmap.cuda()
            z = vae.encode(image.repeat(1, 3, 1, 1)).latent_dist.sample().mul_(0.18215)

            samples = rf.sample(z, sample_steps=10)
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
            plt.suptitle("Height maps generated by Rectified Flow")
            plt.savefig(f"contents/sample_epoch{epoch}.png")
            
            # image sequences to gif
            # gif = []
            # for image in images: 
            #     image = image.clamp(0, 1)
            #     # image = (image - image.min()) / (image.max() - image.min())
            #     x_as_image = make_grid(image.float(), nrow=4)
            #     img = x_as_image.permute(1, 2, 0).cpu().numpy()
            #     img = (img * 255).astype(np.uint8)
            #     gif.append(Image.fromarray(img))

            # gif[0].save(
            #     f"contents/sample_{epoch}.gif",
            #     save_all=True,
            #     append_images=gif[1:],
            #     duration=100,
            #     loop=0,
            # )

            # last_img = gif[-1]
            # last_img.save(f"contents/sample_{epoch}_last.png")
    torch.save(rf.model.state_dict(), 'DiT_XL_2.ckpt')
            