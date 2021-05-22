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
import torch.autograd as autograd
import torch

filepath = os.path.dirname(os.path.realpath(__file__))
os.makedirs("{}/info".format(filepath), exist_ok=True)
os.makedirs("{}/images".format(filepath), exist_ok=True)
os.makedirs("{}/images_for_fid".format(filepath), exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--fid_batch_size", type=int, default=1000, help="size of batches in fid calculation")
parser.add_argument("--update_eps", type=float, default=1e4, help="step size of sample updates")
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


# Loss weight for gradient penalty
lambda_gp = 10

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

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


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

        batch_crt = crt[random.sample(range(crt.size(0)), imgs.size(0))]

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(batch_crt)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, batch_crt.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        # Train the particles every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Particles
            # -----------------
            for p in discriminator.parameters():
                p.requires_grad = False

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            crt = crt.requires_grad_(True)
            fake_validity = discriminator(crt)
            s_loss = -torch.mean(fake_validity)

            s_grad = torch.autograd.grad(s_loss, crt)[0]
            # crt = ((crt - opt.update_eps/(epoch+1)**0.5*s_grad[0]).detach()).clamp(-1.,1.) # + torch.randn_like(crt)*0.5/(epoch+1)**0.2
            crt = (crt - 0.05/(epoch+1)**0.5*s_grad/(s_grad.norm(dim=1,keepdim=True)+1e-6)).detach().clamp(-1.,1.)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [S loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), s_loss.item())
            )

            d_losses.append(d_loss.item())
            s_losses.append(s_loss.item())

            if batches_done % opt.sample_interval == 0:
                save_image(crt.data[random.sample(range(crt.size(0)), 25)], "{}/images/{}.png".format(filepath, batches_done), nrow=5, normalize=True)

            batches_done += opt.n_critic

            if epoch % 20 == 0:
                torch.save(crt.detach(), "{}/images_for_fid/{}.pt".format(filepath, epoch))

info = {'d_losses': d_losses, 's_losses': s_losses}
with open('{}/info/info.pkl'.format(filepath), 'wb') as f:
    pickle.dump(info, f)

# Read pkl file
with open('{}/info/info.pkl'.format(filepath), 'rb') as f:
    info = pickle.load(f)
print(info)



