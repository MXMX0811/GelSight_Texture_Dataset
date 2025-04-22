'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2025-04-22 16:02:06
LastEditors: Mingxin Zhang
LastEditTime: 2025-04-23 02:45:41
Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
'''
import os
import math
import glob
import shutil
import subprocess
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

# Diffusers (Hugging Face)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    DiffusionPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

from huggingface_hub import hf_hub_download, whoami

from peft import LoraConfig, get_peft_model, PeftModel
import cv2

DEVICE = torch.device("cuda")


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
#                                       LoRA                                    #
#################################################################################  
def prepare_lora_model(lora_config, pretrained_model_name_or_path, model_path=None, weight_dtype=torch.bfloat16, resume=False, merge_lora=False):
    """
    (1) 目标:
        - 加载完整的 Stable Diffusion 模型，包括 LoRA 层，并根据需要合并 LoRA 权重。这包括 Tokenizer、噪声调度器、UNet、VAE 和文本编码器。

    (2) 参数:
        - lora_config: LoraConfig, LoRA 的配置对象
        - pretrained_model_name_or_path: str, Hugging Face 上的模型名称或路径
        - model_path: str, 预训练模型的路径
        - resume: bool, 是否从上一次训练中恢复
        - merge_lora: bool, 是否在推理时合并 LoRA 权重

    (3) 返回:
        - noise_scheduler: DDPMScheduler
        - transformer: SD3Transformer2DModel
        - vae: AutoencoderKL
    """
    # 加载噪声调度器，用于控制扩散模型的噪声添加和移除过程
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, cache_dir=model_path, subfolder="scheduler")

    # 加载 VAE 模型，用于在扩散模型中处理图像的潜在表示
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=model_path,
        torch_dtype=weight_dtype,
        subfolder="vae"
    )

    # 加载 UNet 模型，负责处理扩散模型中的图像生成和推理过程
    transformer = SD3Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=model_path,
        torch_dtype=weight_dtype,
        subfolder="transformer"
    )
    
    # 如果设置为继续训练，则加载上一次的模型权重
    if resume:
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("当 resume 设置为 True 时，必须提供有效的 model_path")
        # 使用 PEFT 的 from_pretrained 方法加载 LoRA 模型
        transformer = PeftModel.from_pretrained(transformer, os.path.join(model_path, "transformer"))

        # 确保 transformer 的可训练参数的 requires_grad 为 True
        for param in transformer.parameters():
            if param.requires_grad is False:
                param.requires_grad = True
                
        print(f"✅ 已从 {model_path} 恢复模型权重")

    else:
        # 将 LoRA 配置应用到 text_encoder 和 unet
        transformer = get_peft_model(transformer, lora_config)

        # 打印可训练参数数量
        print("📊 transformer 可训练参数:")
        transformer.print_trainable_parameters()
    
    if merge_lora:
        # 合并 LoRA 权重到基础模型，仅在推理时调用
        transformer = transformer.merge_and_unload()

        # 切换为评估模式
        transformer.eval()

    # 冻结 VAE 参数
    vae.requires_grad_(False)

    # 将模型移动到 GPU 上并设置权重的数据类型
    transformer.to(DEVICE, dtype=weight_dtype)
    vae.to(DEVICE, dtype=weight_dtype)
    
    return noise_scheduler, transformer, vae
    

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    try:
        info = whoami()
        print(f"Authenticated to Huggingface Hub as: {info['name']}")
    except Exception as e:
        raise RuntimeError("You are not logged into Huggingface Hub. Please run 'huggingface-cli login' first.")
    
    # 训练相关参数
    weight_dtype = torch.bfloat16  # 权重数据类型，使用 bfloat16 以节省内存并加快计算速度
    snr_gamma = 5  # SNR 参数，用于信噪比加权损失的调节系数
        
    # Stable Diffusion LoRA 的微调参数

    # 学习率调度器参数
    lr_scheduler_name = "cosine_with_restarts"  # 设置学习率调度器为 Cosine annealing with restarts，逐渐减少学习率并定期重启
    lr_warmup_steps = 100  # 学习率预热步数，在最初的 100 步中逐渐增加学习率到最大值
    max_train_steps = 2000  # 总训练步数，决定了整个训练过程的迭代次数
    num_cycles = 3  # Cosine 调度器的周期数量，在训练期间会重复 3 次学习率周期性递减并重启

    # LoRA 配置
    lora_config = LoraConfig(
        r=32,  # LoRA 的秩，即低秩矩阵的维度，决定了参数调整的自由度
        lora_alpha=16,  # 缩放系数，控制 LoRA 权重对模型的影响
        target_modules=[
            "attn.add_k_proj", 
            "attn.add_q_proj", 
            "attn.add_v_proj", 
            "attn.to_add_out", 
            "attn.to_k", 
            "attn.to_out.0", 
            "attn.to_q", 
            "attn.to_v"
        ],
        lora_dropout=0  # LoRA dropout 概率，0 表示不使用 dropout
    )
    
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
    
    # 准备模型
    noise_scheduler, transformer, vae = prepare_lora_model(
        lora_config,
        "stabilityai/stable-diffusion-3.5-large",
        model_path=args.model_path,
        weight_dtype=weight_dtype,
        resume=False,
        merge_lora=False
    )

    transformer_lora_layers = [p for p in transformer.parameters() if p.requires_grad]
    
    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(transformer_lora_layers, lr=5e-4)

    # 设置学习率调度器
    lr_scheduler = get_scheduler(
        lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=num_cycles
    )

    print("✅ 模型和优化器准备完成！可以开始训练。")
    
    # 禁用并行化，避免警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 初始化
    global_step = 0

    # 进度条显示训练进度
    progress_bar = tqdm(
        range(max_train_steps),  # 根据 num_training_steps 设置
        desc="训练步骤",
    )

    # 训练循环
    for epoch in range(math.ceil(max_train_steps / len(loader))):
        # 如果你想在训练中增加评估，那在循环中增加 train() 是有必要的
        transformer.train()
        
        for i, (input_images, target_images) in enumerate(loader):
            if global_step >= max_train_steps:
                break
            
            input_images = input_images.to(DEVICE, dtype=weight_dtype)
            target_images = target_images.to(DEVICE, dtype=weight_dtype)

            # 2. 编码成latent
            with torch.no_grad():
                input_latents = vae.encode(input_images.repeat(1,3,1,1)).latent_dist.sample() * vae.config.scaling_factor
                target_latents = vae.encode(target_images.repeat(1,3,1,1)).latent_dist.sample() * vae.config.scaling_factor

            # 为潜在表示添加噪声，生成带噪声的图像
            noise = torch.randn_like(target_latents)  # 生成与潜在表示相同形状的随机噪声
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (target_latents.shape[0],), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

            # 计算目标值
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise  # 预测噪声
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_latents, noise, timesteps)  # 预测速度向量

            # transformer 模型预测
            model_pred = transformer(hidden_states=noisy_latents, 
                                     encoder_hidden_states=input_latents, 
                                     pooled_projections=torch.zeros(args.batch_size, 2048, device=DEVICE, dtype=weight_dtype),    # No text encoder, a dummy input 
                                     timestep=timesteps)[0]

            # 计算损失
            if not snr_gamma:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # 计算信噪比 (SNR) 并根据 SNR 加权 MSE 损失
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                
                # 计算加权的 MSE 损失
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # 反向传播
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1

            # 打印训练损失
            if global_step % 100 == 0 or global_step == max_train_steps:
                print(f"🔥 步骤 {global_step}, 损失: {loss.item()}")

            # 保存中间检查点，当前简单设置为每 500 步保存一次
            if global_step % 500 == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)

                # 使用 save_pretrained 保存 PeftModel
                transformer.save_pretrained(os.path.join(save_path, "unet"))
                print(f"💾 已保存中间模型到 {save_path}")

    # 保存最终模型到 checkpoint-last
    save_path = os.path.join(args.output_dir, "checkpoint-last")
    os.makedirs(save_path, exist_ok=True)
    transformer.save_pretrained(os.path.join(save_path, "unet"))
    print(f"💾 已保存最终模型到 {save_path}")

    print("🎉 微调完成！")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="pretrained_models", help="Path to pre-trained SD3.5 model.")
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