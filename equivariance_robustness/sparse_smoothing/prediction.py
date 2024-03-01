import torch.nn.functional as F
from scipy.stats import binom_test
from tqdm.auto import tqdm
import numpy as np
import torch
from .smoothing import sample_graphs


def smoothed_gnn(hparams, model, data, idx, nc, n_samples, batch_size, progress_bar=True):
    votes = torch.zeros(data.x.shape[0], nc, dtype=torch.int32)
    split = torch.ones(n_samples).split(batch_size)
    batch_sizes = [sum(x).int().item() for x in split]

    with torch.no_grad():
        for batch_size in tqdm(batch_sizes, disable=not progress_bar):
            x, edge_idx = sample_graphs(data, hparams, batch_size=batch_size)
            predictions = model(x, edge_idx).argmax(1).cpu()

            predictions = F.one_hot(predictions, int(nc))
            votes += predictions.reshape(batch_size, -1, nc).sum(0)
    return votes[idx.cpu()]


def smoothed_gnn_graph_classification(hparams, model, dataloader, nc, n_samples, progress_bar=True):

    votes = torch.zeros(len(dataloader), nc, dtype=torch.int32)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), disable=not progress_bar):
            data.to(hparams['device'])
            for _ in range(n_samples):
                x, edge_idx = sample_graphs(data, hparams, batch_size=1)
                predictions = model(x, edge_idx, data.batch).argmax(1).cpu()
                votes[i, :] += F.one_hot(predictions[0], int(nc))
    return votes
