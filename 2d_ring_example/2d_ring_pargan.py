### This .py implements the 2D ring-like toy example with ParGAN.

import argparse
import os
import numpy as np
import math
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark')

os.makedirs("parimages", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr_G", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--lr_D", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=2, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval betwen image samples")
parser.add_argument("--update_eps", type=float, default=1e2, help="step size of sample updates")
opt = parser.parse_args()
print(opt)

img_shape = opt.img_size

cuda = True if torch.cuda.is_available() else False

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

def pot2(z):
    # ring-like distribution
    z = z.T
    return (((z[0]**2+z[1]**2)>1.) & ((z[0]**2+z[1]**2)<2.)).float()

N = 1000
Ns = 10000
tgt = torch.rand(Ns,2)*4. - 2.
tgt = tgt * pot2(tgt).reshape(Ns, 1)
idx = torch.where(torch.any(tgt[..., :]!=0, axis=1))[0]
tgt = tgt[idx][:N]

# Initialize generator and discriminator
init = torch.randn(N, int(img_shape))*0.01
crt = init.clone()

plt.figure(figsize=(6, 6))
plt.scatter(crt[:,0],crt[:,1],s=1)
plt.scatter(tgt[:,0],tgt[:,1],s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("parimages/init.png", bbox_inches='tight')
plt.close()

discriminator = Discriminator()

if cuda:
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/2d", exist_ok=True)
dataloader = torch.utils.data.DataLoader(tgt,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

d_losses, s_losses = [], []

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        s_valid = Variable(Tensor(N, 1).fill_(1.0), requires_grad=False)
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        crt = Variable(crt.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        for p in discriminator.parameters():
            p.requires_grad = False

        crt = crt.requires_grad_(True)
        s_loss = adversarial_loss(discriminator(crt), s_valid)
        s_grad = torch.autograd.grad(s_loss, crt)
        crt = (crt - opt.update_eps/(epoch+1)**0.5*s_grad[0]).detach() + torch.randn_like(crt)*0.1/(epoch+1)**0.5

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in discriminator.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        batch_crt = crt[random.sample(range(crt.size(0)), imgs.size(0))]
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(batch_crt.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [S loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), s_loss.item())
        )

        d_losses.append(d_loss.item())
        s_losses.append(s_loss.item())

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            
            plt.figure(figsize=(6, 6))
            plt.scatter(crt[:,0],crt[:,1],s=1)
            plt.scatter(tgt[:,0],tgt[:,1],s=1)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig("parimages/%d.png" % batches_done, bbox_inches='tight')
            plt.close()
