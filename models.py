# -*- coding: utf-8 -*-
# curator/models.py
"""
Pieces of model:

    encoder input:    (..., 3, size, size)
    encoder output 1: (..., n_encode)
    encoder output 2: (..., n_encode)

    decoder input:  (..., n_encode)
    decoder output: (..., 3, size, size)

    classifier input:  (..., 3, size, size)
    classifier output: (..., n_classes)

Default configuration

    size      = 256
    n_encode  = 1024
    n_classes = 1

For reference:

    input:  (...,  c_in,  s_in,  s_in)
    output: (..., c_out, s_out, s_out)

    nn.Conv2d
    s_out = floor( 1 + (s_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride )
    s_out = floor( 1 + (s_in - kernel_size) / stride )

    nn.ConvTranspose2d
    s_out = (s_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    s_out = (s_in - 1) * stride + kernel_size

@version: 2019-12-01
@author: lawortsmann
"""
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    To rank 1+1
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """
    From rank 1+1 to rank 1+3
    """
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAEGAN(nn.Module):
    """
    Variational Autoencoder GAN
    """
    def __init__(self):
        super(VAEGAN, self).__init__()
        ## normalization of image channels
        self.norm_m = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.norm_s = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        ## VAE encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        ## VAE bottleneck
        self.fc_mu = nn.Linear(1024, 1024)
        self.fc_lv = nn.Linear(1024, 1024)
        ## VAE decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            UnFlatten(),
            nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        ## self portrait
        params = []
        for p in self.decoder.parameters():
            params.append( p.flatten() )
        params = torch.cat(params)
        self.psize = np.sqrt(len(params) / 3.0)
        self.psize = 2**int(np.log2(self.psize))
        params = params[:(3 * self.psize * self.psize)]
        self.portrait = params.reshape((1, 3, self.psize, self.psize))
        self.portrait = nn.Sigmoid()(self.portrait)
        self.portrait = (self.portrait - self.norm_m) / self.norm_s
        ## GAN classifier
        self.classifier = torchvision.models.resnet18(pretrained=True)
        ## no gradients
        # for p in self.classifier.parameters():
        #     p.requires_grad = False
        ## add classification output layer
        # self.classifier.fc = nn.Linear(2048, n_classes)
        self.classifier.fc = nn.Linear(512, 1)
        ## some other layers which might be useful
        # nn.MaxPool2d
        # nn.AvgPool2d
        # nn.BatchNorm2d

    def reparameterize(self, mu, logvar):
        """
        Sample from encoded space
        """
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h, sample=True):
        """
        bottleneck layer
        """
        mu, logvar = self.fc_mu(h), self.fc_lv(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        """
        Encode pass
        """
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        """
        Decode pass
        """
        z = self.decoder(z)
        z = (z - self.norm_m) / self.norm_s
        return z

    def sample_portrait(self, n=64, size=64):
        """
        Sample portrait
        """
        affine = torch.rand(n, 2) - 0.5
        resize = (torch.rand(n) + 0.25) / 1.25
        resize = resize * (1 - torch.norm(affine, dim=1))
        theta  = torch.zeros((n, 2, 3))
        theta[:, 0, 0] = resize
        theta[:, 1, 1] = resize
        theta[:, 0, 2] = affine[:, 0]
        theta[:, 1, 2] = affine[:, 1]
        grid = F.affine_grid(theta, size=(n, 3, size, size), align_corners=True)
        crop = torch.repeat_interleave(self.portrait, n, dim=0)
        crop = F.grid_sample(crop, grid, align_corners=True)
        return crop

    def forward(self, x):
        """
        Forward pass
        """
        z, mu, logvar = self.encode(x)
        y = self.decode(z)
        return y, mu, logvar
