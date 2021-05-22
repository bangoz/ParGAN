import argparse
import os, sys
import numpy as np
import math
import pickle
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

filepath = os.path.dirname(os.path.realpath(__file__))
os.makedirs("{}/info".format(filepath), exist_ok=True)
os.makedirs("{}/images".format(filepath), exist_ok=True)
os.makedirs("{}/images_for_fid".format(filepath), exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--fid_batch_size", type=int, default=1000, help="size of batches in fid calculation")
parser.add_argument("--update_eps", type=float, default=3e4, help="step size of sample updates")
opt = parser.parse_args()
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
init = torch.rand(opt.fid_batch_size, int(np.prod(img_shape))) * 2. - 1.
crt = init.clone().view(init.size(0), *img_shape)

discriminator = Discriminator()

if cuda:
    discriminator.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "~/datasets",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

d_losses, s_losses = [], []

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        crt = Variable(crt.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in discriminator.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        # Sample noise as generator input
        batch_crt = crt[random.sample(range(crt.size(0)), imgs.size(0))]

        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(batch_crt))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the particles every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Particles
            # -----------------
            for p in discriminator.parameters():
                p.requires_grad = False

            # Generate a batch of images
            crt = crt.requires_grad_(True)
            # Adversarial loss
            loss_S = -torch.mean(discriminator(crt))
            s_grad = torch.autograd.grad(loss_S, crt)
            crt = (crt - opt.update_eps/(epoch+1)**0.5*s_grad[0]).clamp(-1.,1.).detach()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [S loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_S.item())
            )

            d_losses.append(loss_D.item())
            s_losses.append(loss_S.item())

        if batches_done % opt.sample_interval == 0:
            save_image(crt.data[random.sample(range(crt.size(0)), 25)], "{}/images/{}.png".format(filepath, batches_done), nrow=5, normalize=True)
        batches_done += 1

        if epoch % 20 == 0:
            torch.save(crt.detach(), "{}/images_for_fid/{}.pt".format(filepath, epoch))

info = {'d_losses': d_losses, 's_losses': s_losses}
with open('{}/info/info.pkl'.format(filepath), 'wb') as f:
    pickle.dump(info, f)

# Read pkl file
with open('{}/info/info.pkl'.format(filepath), 'rb') as f:
    info = pickle.load(f)
print(info)
