import numpy as np
import argparse
import os, sys
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

class Discriminator(nn.Module):
    def __init__(self, dim, n_layer=2, n_neuron=128):
        """
        dim: input dim
        n_layer: number of FC layers
        n_neuron: number of neurons in each FC layer
        """
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.n_neuron = n_neuron
        
        model = [nn.Linear(self.dim, self.n_neuron), nn.ReLU()]
        for i in range(self.n_layer-1):
            model.append(nn.Linear(self.n_neuron, self.n_neuron))
            model.append(nn.ReLU())
        model.append(nn.Linear(self.n_neuron, 1))
        model.append(nn.Sigmoid())
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


class ParGAN():
    def __init__(self, lr, b1, b2, ref):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.adversarial_loss = torch.nn.BCELoss()
        self.ref = ref
 
    def update(self, D, x0, lnref, lnprob, n_iter = 1000, batchsize=100, stepsize = 1e-3, alpha = 0.9, debug = False):
        # Check input
        assert self.ref.shape[0] == x0.shape[0] == batchsize # full batch
        if x0 is None or lnprob is None or lnref is None:
            raise ValueError('x0 or lnprob or lnref cannot be None!')
        
        theta = x0.copy()

        disc = Discriminator(D)
        optimizer_D = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        valid = Variable(torch.FloatTensor(batchsize, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(batchsize, 1).fill_(0.0), requires_grad=False)

        tensor_theta = torch.FloatTensor(theta)
        tensor_ref = torch.FloatTensor(self.ref)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for itr in range(n_iter):
            if debug and (itr+1) % 1000 == 0:
                print('iter ' + str(itr+1))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for p in disc.parameters():
                p.requires_grad = True

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.adversarial_loss(disc(tensor_ref), valid)
            fake_loss = self.adversarial_loss(disc(tensor_theta), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()


            # ---------------------
            #  Train Particles
            # ---------------------
            for p in disc.parameters():
                p.requires_grad = False

            lnpgrad = lnprob(tensor_theta.numpy())
            lnrefgrad = lnref(tensor_theta.numpy())
            # calculating the kernel matrix
            tensor_theta = tensor_theta.requires_grad_(True)
            # loss = torch.log(1/disc(tensor_theta)-1).sum()
            loss = -torch.log(disc(tensor_theta)+1e-10).sum()
            grad = torch.autograd.grad(loss, tensor_theta)[0]

            grad_theta = grad.numpy() + lnrefgrad - lnpgrad # numpy
            
            # adagrad 
            if itr == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            tensor_theta = tensor_theta.requires_grad_(False) - torch.FloatTensor(stepsize * adj_grad)
            
        return tensor_theta.numpy()
    
