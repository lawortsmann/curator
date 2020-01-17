# -*- coding: utf-8 -*-
# movies.py
"""
A simple RNN for modeling movie scripts.

@version: 2020-01-15
@author: lawortsmann
"""
import numpy as np
import pandas as pd
from six.moves.urllib import request
from time import sleep
import os, re
import torch
from torch import nn

## functions to import data from imsdb.com

def parse_script_text(data):
    """
    Parse html from imsdb.com to extract the script text
    """
    start_match = b'<td class="scrtext">'
    stop_match = b'\n<table'
    i = data.find(start_match) + len(start_match)
    j = data[i:].find(stop_match)
    txt = data[i:][:j].decode(errors='ignore')
    txt = re.sub(r'<script>.*</script>', '', txt, flags=re.DOTALL)
    txt = re.sub(r'\<[^<>]*\>', '', txt, flags=re.DOTALL)
    txt = txt.replace('\r', '').strip()
    return txt


def import_scripts(scripts, data_dir="scripts/", wait=True):
    """
    Import scripts from imsdb and save to `data_dir`
    """
    base_url = "https://www.imsdb.com/scripts/"
    for script in scripts:
        print( script )
        ## wait
        if wait:
            sleep(1.0)
        ## download
        with request.urlopen(base_url + script + ".html") as page:
            data = page.read()
        ## parse
        text = parse_script_text(data)
        ## save
        with open(data_dir + script + '.txt', 'w') as file:
            file.write(text)
    return True

## functions to load downloaded scripts

def tokenizer(data):
    """
    Convert script text to tokens
    """
    text = " <p> ".join( data.split('\n\n') )
    text = " ".join(text.split())
    text = re.sub('[^\w\'<>]', ' ', text.lower())
    return text.split()


def load_scripts(data_dir='scripts/', tokenize=True):
    """
    Load script from `data_dir` and tokenize
    """
    scripts = [fn for fn in os.listdir(data_dir) if '.txt' in fn]
    dataset = dict()
    for script in scripts:
        with open(data_dir + script, 'r') as file:
            data = file.read()
        script = script.split('.txt')[0]
        if tokenize:
            dataset[script] = tokenizer(data)
        else:
            dataset[script] = data
    return dataset


def build_vocab(dataset, min_count=1):
    """
    Build Vocabulary
    """
    vocab, counts = np.unique(sum(dataset.values(), []), return_counts=True)
    vocab = vocab[counts > min_count]
    index = np.arange(len(vocab))
    vocab = pd.Series(index, index=vocab)
    res = dict()
    for key, val in dataset.items():
        idx = vocab.reindex(val, fill_value=-1)
        res[key] = np.array(idx, dtype=int)
    return vocab, res


def corpus_invfreq(dataset, sqrt=True):
    """
    Inverse frequency weights
    """
    corpus = np.concatenate(list(dataset.values()))
    corpus = corpus[corpus >= 0]
    ix, ct = np.unique(corpus, return_counts=True)
    w = np.ones(max(corpus) + 1)
    if sqrt:
        w[ix] = np.sqrt(np.mean(ct) / ct)
    else:
        w[ix] = np.mean(ct) / ct
    w = w / np.mean(w)
    return w


def dataset_generator(dataset, n_batch=64, n_seq=256):
    """
    Build the dataset pipeline
    """
    seq = n_seq - np.arange(n_seq)
    keys = list(dataset.keys())
    while True:
        docs = np.random.choice(keys, size=n_batch)
        x = np.zeros((n_seq, n_batch), dtype=int)
        y = np.zeros(n_batch, dtype=int)
        for i, doc in enumerate(docs):
            idx = dataset[doc]
            idx = idx[idx >= 0]
            jx = np.random.randint(low=n_seq, high=len(idx))
            ix = jx - seq
            x[:, i] = idx[ix]
            y[i] = idx[jx]
        yield docs, x, y
    return True

## Simple RNN model

class SimpleRNN(nn.Module):
    """
    Simple RNN
    """
    def __init__(self, n_vocab, n_embed, n_hidden, n_layers=1, dropout=0.0):
        super(SimpleRNN, self).__init__()
        ## dimensionality
        self.n_vocab = int(n_vocab)
        self.n_embed = int(n_embed)
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        ## embedding
        self.embed = nn.Embedding(self.n_vocab, self.n_embed)
        ## forward LSTM
        kwargs = dict(num_layers=self.n_layers, dropout=dropout)
        self.rnn = nn.LSTM(self.n_embed, self.n_hidden, **kwargs)
        self.lin = nn.Linear(self.n_hidden, self.n_vocab)
        self.act = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass
        """
        ## get embedding
        y = self.embed(x)
        ## lstm layers
        _, (z, _) = self.rnn(y)
        ## linear layer
        z = self.lin(z[0])
        z = self.act(z)
        return z
    
    
def training(model, pipeline, w=None, lr=0.01, n_epochs=32, n_steps=64):
    """
    Training...
    """
    ## initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ## initialize loss function
    if w is None:
        nll_loss = nn.NLLLoss()
    else:
        w = torch.tensor( np.array(w, dtype=np.float32) )
        nll_loss = nn.NLLLoss(weight=w)
    ## training loop
    logs = []
    for epoch in range(n_epochs):
        print( "Epoch: %i"%(epoch + 1) )
        for i, (docs, x, y) in enumerate(pipeline):
            ## convert to tensors
            x = torch.tensor(x)
            y = torch.tensor(y)
            ## forward pass
            z = model(x)
            loss = nll_loss(z, y)
            ## step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## store for logging
            logs.append({
                'epoch': epoch,
                'batch': i,
                'loss': float(loss),
            })
            ## stop
            if (i + 1) >= n_steps:
                break
    ## return model and logs
    logs = pd.DataFrame(logs)
    return model, logs


if __name__ == "__main__":
    from argparse import ArgumentParser
    np.warnings.simplefilter(action='ignore')
    
    ## arguments
    parser = ArgumentParser(description="Run the Simple RNN Model")
    parser.add_argument('--verbose', action='store_true', default=False, help='display status')
    args = parser.parse_args()
    
    ## import data
    import_scripts(scripts, data_dir="scripts/")
    
    ## load data
    scripts = load_scripts(data_dir="scripts/", tokenize=True)
    vocab, dataset = build_vocab(scripts, min_count=1)
    pipeline = dataset_generator(dataset, n_batch=64, n_seq=64)
    
    ## initialize model
    model = SimpleRNN(len(vocab), n_embed=64, n_hidden=64, n_layers=4, dropout=0.25)
    
    ## train model
    w = corpus_invfreq(dataset, sqrt=True)
    model, logs = training(model, pipeline, w=w, lr=0.1, n_epochs=16, n_steps=64)
    hist = logs.groupby('epoch').mean()
