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
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--fid_batch_size", type=int, default=1000, help="size of batches in fid calculation")
parser.add_argument("--update_eps", type=float, default=1e3, help="step size of sample updates")
opt = parser.parse_args()
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
init = torch.rand(opt.fid_batch_size, int(np.prod(img_shape))) * 2. - 1.
crt = init.clone().view(init.size(0), *img_shape)

discriminator = Discriminator()

if cuda:
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
discriminator.apply(weights_init_normal)

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "~/datasets",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

d_losses, s_losses = [], []

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        s_valid = Variable(Tensor(opt.fid_batch_size, 1).fill_(1.0), requires_grad=False)
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        crt = Variable(crt.type(Tensor))

        # -----------------
        #  Train Particles
        # -----------------
        for p in discriminator.parameters():
            p.requires_grad = False

        crt = crt.requires_grad_(True)
        s_loss = adversarial_loss(discriminator(crt), s_valid)
        s_grad = torch.autograd.grad(s_loss, crt)
        crt = (crt - opt.update_eps/(epoch+1)**0.5*s_grad[0]).clamp(-1.,1.).detach()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in discriminator.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        batch_crt = crt[random.sample(range(crt.size(0)), imgs.size(0))]
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(batch_crt), fake)
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
            save_image(crt.data[random.sample(range(crt.size(0)), 25)], "{}/images/{}.png".format(filepath, batches_done), nrow=5, normalize=True)
        
        if epoch % 20 == 0:
            torch.save(crt.detach(), "{}/images_for_fid/{}.pt".format(filepath, epoch))

info = {'d_losses': d_losses, 's_losses': s_losses}
with open('{}/info/info.pkl'.format(filepath), 'wb') as f:
    pickle.dump(info, f)

# Read pkl file
with open('{}/info/info.pkl'.format(filepath), 'rb') as f:
    info = pickle.load(f)
print(info)
