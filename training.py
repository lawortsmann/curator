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
from models import VAEGAN, vae_loss
## pytorch
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def vae_training(model, pipeline, original=None, verbose=True, n_epochs=64, n_steps=256):
    ## optimizer just for VAE parmeters
    optimizer = torch.optim.Adam(model.parameters())
    ## training loop
    logs = []
    message = "Epoch %i [%s%s] %i/%i\r"
    for epoch in range(n_epochs):
        for i, (batch, _) in enumerate(pipeline):
            ## done with epoch
            if i >= n_steps:
                break
            ## forward pass
            y, mu, logvar = model(batch)
            ## evaluate loss function
            loss, err, kld = vae_loss(y, batch, mu, logvar)
            ## step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## store for logging
            logs.append({
                'epoch': epoch,
                'batch': i,
                'loss': float(loss),
                'err':  float(err),
                'kld':  float(kld)
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
            ## get normalized reconstruction
            tm = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            ts = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            recong, _, _ = model(original)
            recong = (recong.detach() - tm) / ts
            ## plot
            fig = plot_batch(recong, size=64, fname="epoch_%i.png"%epoch)
    ## return model and logs
    logs = pd.DataFrame(logs)
    return model, logs


def main():
    ## get data pipeline
    pipeline = build_pipeline('jjjjound/', size=64)
    for original, _ in pipeline: break
    fig = plot_batch(original, size=64, fname="original.png")

    ## setup model
    model = VAEGAN()

    ## VAE training...
    model, logs = vae_training(model, pipeline, original=original, verbose=True)

    ## save
    logs.to_csv('logs.csv')
    with np.warnings.catch_warnings(record='ignore'):
        torch.save(model, 'model.pt')
    return True


if __name__ == "__main__":
    main()
