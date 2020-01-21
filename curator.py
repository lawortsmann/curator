# -*- coding: utf-8 -*-
# curator/curator.py
"""
For reference:

    input:  (...,  c_in,  s_in,  s_in)
    output: (..., c_out, s_out, s_out)

    nn.Conv2d
    s_out = floor( 1 + (s_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride )
    s_out = floor( 1 + (s_in - kernel_size) / stride )

    nn.ConvTranspose2d
    s_out = (s_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    s_out = (s_in - 1) * stride + kernel_size

@version: 2020-01-20
@author: lawortsmann
"""
import numpy as np
import pandas as pd
from sys import stdout
import json, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader
from torchvision import transforms, datasets


class UnFlatten(nn.Module):
    """
    From rank 1+1 to rank 1+3
    """
    def forward(self, input):
        new_shape = tuple(input.shape) + (1, 1)
        return input.view(*new_shape)


class CVAE(nn.Module):
    """
    Convolutional Variational Autoencoder
    """
    def __init__(self):
        super(CVAE, self).__init__()
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
            nn.Flatten()
        )
        ## VAE bottleneck
        self.lin_mu = nn.Linear(1024, 64)
        self.lin_lv = nn.Linear(1024, 64)
        ## VAE decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        Encode pass
        """
        ## evaluate encoder
        h = self.encoder(x)
        z_mu = self.lin_mu(h)
        z_lv = self.lin_lv(h)
        ## sample from encoded space
        std = torch.exp(z_lv / 2.0)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(z_mu)
        return z, z_mu, z_lv

    def decode(self, z):
        """
        Decode pass
        """
        ## evaluate decoder
        y = self.decoder(z)
        return y
    
    def forward(self, x, sample=True):
        """
        Forward pass
        """
        ## encode
        z, z_mu, z_lv = self.encode(x)
        ## decode
        if sample:
            y = self.decode(z)
        else:
            y = self.decode(z_mu)
        return y, z_mu, z_lv


def build_pipeline(datadir, size=64, n_batch=256, batches=256, n_workers=2):
    """
    Build an image dataset pipeline for images in subfolders of `datadir`
    """
    ## image transforms
    process_patch = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.1, 1.0)),
        transforms.ToTensor(),
    ])
    ## image dataset
    images = datasets.ImageFolder(root=datadir, transform=process_patch)
    ## DataLoader
    num_samples = (1 + batches) * n_batch
    sampler = RandomSampler(images, replacement=True, num_samples=num_samples)
    kwargs = dict(sampler=sampler, batch_size=n_batch, num_workers=n_workers, drop_last=True)
    pipeline = DataLoader(images, **kwargs)
    ## return pipeline
    return pipeline


def plot_batch(batch, title=None, fname=None, size=256, n_col=8, n_row=8):
    """
    Plot an image batch
    """
    import matplotlib.pyplot as plt

    ## inverse transform image patches
    images = []
    for i, img in enumerate(batch):
        img = img.numpy().T.swapaxes(0, 1)
        img = np.round(255 * img).astype(np.uint8)
        images.append( img )
    ## manipulate image array
    grid_shape = (n_col, n_row, size, size, 3)
    images = np.reshape(images, grid_shape)
    show_shape = (n_col * size, n_row * size, 3)
    images = images.swapaxes(1, 2).reshape(show_shape)
    ## plot batch
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(images)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if fname is not None:
        plt.imsave(fname, images)
    return fig


def vae_loss(x, y, z_mu, z_lv):
    """
    Variational Autoencoder loss
    """
    ## error of autoencoder
    bce = F.binary_cross_entropy(y, x, reduction='none')
    bce = torch.sum(bce)
    ## Kullback–Leibler divergence of encoding space
    kld = 0.5 * (1 + z_lv - z_mu * z_mu - z_lv.exp())
    kld = torch.sum(kld)
    ## Loss
    loss = bce - kld
    return loss, bce, kld


def training(model, pipeline, lr=0.01, n_epochs=32, verbose=True):
    """
    Training...
    """
    ## set number of threads
    torch.set_num_threads(5)
    ## initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ## training loop
    logs, message = [], "Epoch %i [%s%s] %0.4f \r"
    for epoch in range(n_epochs):
        n_steps = len(pipeline)
        for i, (x, _) in enumerate(pipeline):
            ## clear gradients
            optimizer.zero_grad()
            ## forward pass
            y, z_mu, z_lv = model(x)
            loss, bce, kld = vae_loss(x, y, z_mu, z_lv)
            ## step optimizer
            loss.backward()
            optimizer.step()
            ## store for logging
            logs.append({
                'epoch': epoch,
                'batch': i,
                'loss': float(loss),
                'bce': float(bce),
                'kld': float(kld),
            })
            ## display progress
            if verbose:
                avg_loss = np.mean([h['loss'] for h in logs if h['epoch'] == epoch])
                fill = int(round(50 * (i + 1) / n_steps))
                mssg = message%(
                    epoch + 1, fill * "=", (50 - fill) * " ", avg_loss
                )
                stdout.write(mssg)
                stdout.flush()
        ## out of inner loop
        if verbose:
            avg_loss = np.mean([h['loss'] for h in logs if h['epoch'] == epoch])
            print( message%(epoch + 1, 50 * "=", "", avg_loss) )
    ## out of outer loop
    logs = pd.DataFrame(logs)
    return logs
