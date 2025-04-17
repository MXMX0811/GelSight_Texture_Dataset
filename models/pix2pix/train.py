from __future__ import print_function
import argparse
import os
import pickle
import numpy as np
from math import log10
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate


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


def main(opt):
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    root_dir = "../../Texture"
    dataset = TextureDataset(root_dir, transform=transform)

    training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    print('===> Building models')
    net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        # train
        for iteration, batch in enumerate(training_data_loader, 1):
            # forward
            real_a, real_b = batch[0].to(device), batch[1].to(device)
            fake_b = net_g(real_a)

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()
            
            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)
            
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
        
            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
            
            loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()

            optimizer_g.step()

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # test
        # avg_psnr = 0
        # for batch in testing_data_loader:
        #     input, target = batch[0].to(device), batch[1].to(device)

        #     prediction = net_g(input)
        #     mse = criterionMSE(prediction, target)
        #     psnr = 10 * log10(1 / mse.item())
        #     avg_psnr += psnr
        # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

        #checkpoint
        if epoch % 50 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))
            net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
            net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
            torch.save(net_g, net_g_model_out_path)
            torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    opt = parser.parse_args()

    print(opt)
    main(opt)