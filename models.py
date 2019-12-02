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


def vae_loss(y, x, mu, logvar):
    ## reverse normalize batch
    tm = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    ts = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    xp = x * ts + tm
    ## Autoencoding error
    err = F.binary_cross_entropy(y, xp, reduction='mean')
    # err = F.mse_loss(y, xp, reduction='sum')
    ## Kullback–Leibler divergence of encoding space
    kld = -0.5 * torch.mean(1 + logvar - mu * mu - logvar.exp())
    ## Loss
    loss = err + kld
    return loss, err, kld


class VAEGAN(nn.Module):
    """
    Variational Autoencoder GAN
    """
    def __init__(self):
        super(VAEGAN, self).__init__()
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
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        ## VAE decoder
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        ## GAN classifier
        self.classifier = torchvision.models.resnet18(pretrained=True)
        ## no gradients
        for p in self.classifier.parameters():
            p.requires_grad = False
        ## add classification output layer
        # self.classifier.fc = nn.Linear(2048, n_classes)
        self.classifier.fc = nn.Linear(512, 1)

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
        mu, logvar = self.fc1(h), self.fc2(h)
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
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        """
        Forward pass
        """
        z, mu, logvar = self.encode(x)
        y = self.decode(z)
        return y, mu, logvar
