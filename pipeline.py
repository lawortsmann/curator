# -*- coding: utf-8 -*-
# curator/pipeline.py
"""
Image dataset pipeline.

@version: 2019-11-21
@author: lawortsmann
"""
import numpy as np
import pandas as pd
from torch.utils.data import RandomSampler, DataLoader
from torchvision import transforms, datasets
TMEAN = [0.485, 0.456, 0.406]
TSTD  = [0.229, 0.224, 0.225]


def build_pipeline(datadir, size=256, n_batch=64, n_workers=2):
    """
    Build an image dataset pipeline for images in subfolders of `datadir`
    """
    ## image transforms
    process_patch = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(TMEAN, TSTD)
    ])
    ## image dataset
    images = datasets.ImageFolder(root=datadir, transform=process_patch)
    ## DataLoader
    sample = RandomSampler(images, replacement=True, num_samples=100000)
    kwargs = dict(sampler=sample, batch_size=n_batch, num_workers=n_workers, drop_last=True)
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
        img = 255 * (img * TSTD + TMEAN)
        img = np.round(img).astype(np.uint8)
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
