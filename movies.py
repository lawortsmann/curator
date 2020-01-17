# -*- coding: utf-8 -*-
# movies.py
"""
A simple RNN for modeling movie scripts.

@version: 2020-01-16
@author: lawortsmann
"""
import numpy as np
import pandas as pd
from six.moves.urllib import request
from sys import stdout
from time import sleep
import json, os, re
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
        ## wait
        if wait:
            sleep(1.0)
        try:
            ## download
            with request.urlopen(base_url + script + ".html") as page:
                data = page.read()
            ## parse
            text = parse_script_text(data)
            assert len(text) >= 100
            ## save
            with open(data_dir + script + '.txt', 'w') as file:
                file.write(text)
        except Exception as err:
            print( "Error:" )
            print( script, err )
            print( "======" )
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


def build_vocab(dataset, term_freq=2, doc_freq=2, missing_token='<m>'):
    """
    Build the vocabulary from the document dataset
    """
    ## loop over documents to build full vocabulary
    full_vocab = dict()
    for i, (key, val) in enumerate(dataset.items()):
        w, c = np.unique(val, return_counts=True)
        full_vocab[key] = pd.Series(c, index=w)
    full_vocab = pd.DataFrame(full_vocab)
    full_vocab = full_vocab.fillna(0).astype(int)
    ## filter vocabulary
    g = (np.sum(full_vocab, axis=1) >= term_freq)
    g = g & (np.sum(full_vocab > 0, axis=1) >= doc_freq)
    vocab = 1 * full_vocab[g]
    ## count missing
    if missing_token is not None:
        vocab.loc[missing_token] = np.sum(full_vocab[~g], axis=0)
    ## compute frequency weights
    weights = np.sum(vocab, axis=1).sort_values(ascending=False)
    weights = 1 / (1 + weights)
    weights = weights / np.mean(weights)
    ## build index and lookup
    vocab = 1 * vocab.loc[weights.index]
    index = np.arange(len(vocab))
    index = pd.Series(index, index=vocab.index)
    lookup = pd.Series(index.index, index=index.values)
    return vocab, weights, index, lookup


def build_corpus(dataset, index, missing_token='<m>'):
    """
    Build the corpus from an index and a document dataset
    """
    ## get missing token
    if missing_token in index:
        na_ix = index[missing_token]
    else:
        na_ix = -1
    ## build corpus
    corpus = dict()
    for i, (key, val) in enumerate(dataset.items()):
        ix = index.reindex(val, fill_value=na_ix)
        ix = np.array(ix, dtype=int)
        ix = ix[ix >= 0]
        if len(ix) >= 16:
            corpus[key] = ix
    return corpus


def build_pipeline(corpus, n_seq=32, n_batch=1024):
    """
    Build the training pipeline
    """
    ## get lengths
    lengths = pd.Series({key: len(val) for key, val in corpus.items()})
    L = np.array(lengths, dtype=int)
    ## shape corpus into rectangular array
    n_docs = len(corpus)
    X = np.zeros((n_docs, max(lengths)), dtype=int)
    for i, (key, j) in enumerate(lengths.iteritems()):
        X[i, :j] = corpus[key]
    ## get sequence indexer
    seq = n_seq - np.arange(n_seq)
    ## continuously yield batches
    while True:
        ix = np.random.choice(n_docs, size=n_batch)
        jx = np.random.uniform(low=n_seq, high=L[ix])
        jx = np.array(jx, dtype=int)
        x_i = X[ix, jx - seq[:, None]]
        y_i = X[ix, jx]
        yield ix, x_i, y_i
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


def training(model, pipeline, weights, lr=0.01, n_epochs=32, n_steps=64, verbose=True):
    """
    Training...
    """
    ## set number of threads
    torch.set_num_threads(10)
    ## initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ## initialize loss function
    w = torch.tensor( np.array(weights, dtype=np.float32) )
    nll_loss = nn.NLLLoss(weight=w)
    ## training loop
    logs, message = [], "Epoch %i [%s%s] %0.4f \r"
    for epoch in range(n_epochs):
        for i, (d, x, y) in enumerate(pipeline):
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
            ## display progress
            if verbose:
                avg_loss = np.mean([h['loss'] for h in logs if h['epoch'] == epoch])
                fill = int(round(50 * (i + 1) / n_steps))
                mssg = message%(
                    epoch + 1, fill * "=", (50 - fill) * " ", avg_loss
                )
                stdout.write(mssg)
                stdout.flush()
            ## next epoch
            if (i + 1) >= n_steps:
                break
        ## out of inner loop
        if verbose:
            avg_loss = np.mean([h['loss'] for h in logs if h['epoch'] == epoch])
            print( message%(epoch + 1, 50 * "=", "", avg_loss) )
    ## out of outer loop
    logs = pd.DataFrame(logs)
    return logs


def save_model(vocab, model, metadata, logs, save_dir='movie_run/'):
    ## ensure save_dir exists and is clean
    if os.path.exists(save_dir):
        os.rmdir(save_dir)
    os.mkdir(save_dir)
    ## save vocab
    vocab = pd.DataFrame(vocab)
    vocab.to_csv(save_dir + 'vocab.csv')
    ## save model
    with np.warnings.catch_warnings(record='ignore'):
        torch.save(model, save_dir + 'model.pt')
    ## save metadata
    with open(save_dir + 'metadata.json', 'w'):
        json.dump(metadata, file)
    ## save logs
    logs = pd.DataFrame(logs)
    logs.to_csv(save_dir + 'logs.csv', index=False)
    return save_dir


if __name__ == "__main__":
    from argparse import ArgumentParser
    np.warnings.simplefilter(action='ignore')
    
    ## arguments
    parser = ArgumentParser(description="Simple RNN Model for Text Prediction")
    parser.add_argument('--term_freq', metavar='', type=int, default=32, help='minimum term count')
    parser.add_argument('--doc_freq', metavar='', type=int, default=4, help='minimum doc count')
    parser.add_argument('--data-dir', metavar='', type=str, default='scripts/', help='data directory')
    parser.add_argument('--save-dir', metavar='', type=str, default='movie_run/', help='save directory')
    parser.add_argument('--embed', metavar='', type=int, default=256, help='embedding dimension')
    parser.add_argument('--hidden', metavar='', type=int, default=256, help='hidden dimension')
    parser.add_argument('--layers', metavar='', type=int, default=4, help='number of layers')
    parser.add_argument('--dropout', metavar='', type=float, default=0.05, help='dropout rate between layers')
    parser.add_argument('--learning-rate', metavar='', type=float, default=1.0, help='learning rate')
    parser.add_argument('--seq', metavar='', type=int, default=32, help='sequence length')
    parser.add_argument('--batch', metavar='', type=int, default=1024, help='batch size')
    parser.add_argument('--steps', metavar='', type=int, default=256, help='steps per epoch')
    parser.add_argument('--epochs', metavar='', type=int, default=32, help='number of epochs')    
    parser.add_argument('--missing', action='store_true', help='use a missing token')
    parser.add_argument('--verbose', action='store_true', help='display status')
    args = parser.parse_args()
    
    if args.missing:
        missing_token = '<m>'
    else:
        missing_token = None
    
    print( "Loading Dataset..." )
    ## load data and build vocabulary
    scripts = load_scripts(args.data_dir, tokenize=True)
    vocab, weights, index, lookup = build_vocab(scripts, args.term_freq, args.doc_freq, missing_token)
    ## build corpus
    corpus = build_corpus(scripts, index, missing_token)
    ## build training pipeline
    pipeline = build_pipeline(corpus, n_seq=args.seq, n_batch=args.batch)
    print( "Using %i Documents and %i Words"%(len(corpus), len(weights)) )
    
    print( "Initializing Model..." )
    ## initialize model
    config = {
        "n_vocab": len(vocab),
        "n_embed": args.embed,
        "n_hidden": args.hidden,
        "n_layers": args.layers,
        "dropout": args.dropout,
    }
    model = SimpleRNN(**config)
    ## count parameters
    n_params = sum(np.prod(p.shape) for p in model.parameters())
    print( "Using Model with %i Parameters"%n_params )
    
    ## train model
    print( "Training..." )
    kwargs = dict(n_epochs=args.epochs, n_steps=args.steps, verbose=args.verbose)
    logs = training(model, pipeline, weights, lr=args.learning_rate, **kwargs)
    
    ## save model
    print( "Saving Model..." )
    vocab['INDEX'] = index
    metadata = dict(**config)
    metadata['term_freq'] = args.term_freq
    metadata['doc_freq'] = args.doc_freq
    metadata['n_seq'] = args.seq
    metadata['n_batch'] = args.batch
    metadata['learning_rate'] = args.learning_rate
    path = save_model(vocab, model, metadata, logs, args.save_dir)
    print( "Model Saved to: %s"%path )
