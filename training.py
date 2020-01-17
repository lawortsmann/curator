# -*- coding: utf-8 -*-
# curator/training.py
"""
@version: 2019-12-01
@author: lawortsmann
"""
import numpy as np
import pandas as pd
from sys import stdout
import matplotlib.pyplot as plt
from pipeline import build_pipeline, plot_batch
from models import VAEGAN
## pytorch
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
## globals
SAVEDIR = "run0/"


def vae_loss(y, x, mu, logvar):
    ## Autoencoding error
    err = F.mse_loss(y, x, reduction='mean')
    ## Kullback–Leibler divergence of encoding space
    kld = 0.5 * torch.mean(1 + logvar - mu * mu - logvar.exp())
    ## Loss
    loss = err - kld
    return loss


def gan_loss(p_x, p_y):
    c_x = torch.ones(p_x.shape)
    c_y = torch.zeros(p_y.shape)
    l_x = F.binary_cross_entropy_with_logits(p_x, c_x, reduction='mean')
    l_y = F.binary_cross_entropy_with_logits(p_y, c_y, reduction='mean')
    loss = (l_x + l_y) / 2.0
    return loss


def reg_loss(p_p):
    c_p = torch.ones(p_p.shape)
    l_p = F.binary_cross_entropy_with_logits(p_p, c_p, reduction='mean')
    return l_p


def vae_training(model, pipeline, original=None, verbose=True, n_epochs=64, n_steps=256):
    ## optimizer just for VAE parmeters
    optimizer = torch.optim.Adam(model.parameters())
    ## training loop
    logs = []
    message = "Epoch %i [%s%s] %i/%i\r"
    for epoch in range(n_epochs):
        for i, (x, _) in enumerate(pipeline):
            ## done with epoch
            if i >= n_steps:
                break
            ## forward pass
            y, mu, logvar = model(x)
            p = model.sample_portrait()
            ## evaluate classifier
            p_x = model.classifier(x)
            p_y = model.classifier(y)
            p_p = model.classifier(p)
            ## evaluate loss functions
            l_v = vae_loss(y, x, mu, logvar)
            l_g = gan_loss(p_x, p_y)
            l_p = reg_loss(p_p)
            loss = l_v + l_g - 0.1 * l_p
            ## step optimizer
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            ## store for logging
            logs.append({
                'epoch': epoch,
                'batch': i,
                'loss': float(loss),
                'loss_vae': float(l_v),
                'loss_gan': float(l_g),
                'loss_reg': float(l_p),
            })
            ## verbose logging
            if verbose:
                fill = int(round(50 * (i + 1) / n_steps))
                mssg = message%(
                    epoch, fill * "=", (50 - fill) * " ", i + 1, n_steps
                )
                stdout.write(mssg)
                stdout.flush()
        ## verbose logging
        if verbose:
            print( "" )
        ## plot reconstruction
        if original is not None:
            ## plot reconstruction
            recong, _, _ = model(original)
            f_n = SAVEDIR + "epoch_%i.png"%epoch
            fig = plot_batch(recong.detach(), size=64, fname=f_n)
            plt.close()
            ## save portrait
            portrait = model.norm_s * model.portrait + model.norm_m
            portrait = np.round(255 * portrait.detach().numpy()).astype(np.uint8)
            portrait = portrait[0].swapaxes(0, 2).swapaxes(1, 0)
            plt.imsave(SAVEDIR + "portrait_%i.png"%epoch, portrait)
    ## return model and logs
    logs = pd.DataFrame(logs)
    return model, logs


def main():
    ## set number of threads
    torch.set_num_threads(10)

    ## get data pipeline
    pipeline = build_pipeline('jjjjound/', size=64)
    for original, _ in pipeline: break
    f_n = SAVEDIR + "original.png"
    fig = plot_batch(original, size=64, fname=f_n)
    plt.close()

    ## setup model
    model = VAEGAN()

    ## VAE training...
    model, logs = vae_training(model, pipeline, original=original, verbose=True, n_epochs=32, n_steps=512)

    ## save
    logs.to_csv(SAVEDIR + 'logs.csv', index=False)
    with np.warnings.catch_warnings(record='ignore'):
        torch.save(model, SAVEDIR + 'model.pt')
    return True


if __name__ == "__main__":
    main()
