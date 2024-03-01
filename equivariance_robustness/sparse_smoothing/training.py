import torch
import torch.nn.functional as F

import numpy as np
import logging
from tqdm.auto import tqdm

from .smoothing import sample_graphs


def training_node_classification(model, data, idx_train, idx_valid, hparams):

    if 'early_stopping' in hparams:
        early_stopping = hparams['early_stopping']
    else:
        early_stopping = np.inf

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],
                                 weight_decay=hparams["weight_decay"])

    best_loss = np.inf
    best_epoch = 0
    best_state = {}

    for epoch in range(hparams["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        loss_train = loss_node_classification(hparams, model, data, idx_train)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_val = loss_node_classification(
                hparams, model, data, idx_valid)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

            if hparams["logging"]:
                logging.info(f'Epoch {epoch:4}: '
                             f'loss_train: {loss_train.item():.5f}, '
                             f'loss_val: {loss_val.item():.5f} ')

        if epoch - best_epoch > early_stopping:
            if hparams["logging"]:
                logging.info(f"early stopping at epoch {epoch}")
            break

    if hparams["logging"]:
        logging.info('best_epoch', best_epoch)
    model.load_state_dict(best_state)
    return model.eval()


def loss_node_classification(hparams, model, data, idx):
    x, edge_idx = sample_graphs(data, hparams, batch_size=1)
    logits = model(x, edge_idx)
    return F.cross_entropy(logits[idx], data.y[idx])


def training_graph_classification(model, data_loader_train, data_loader_valid, hparams):

    if 'early_stopping' in hparams:
        early_stopping = hparams['early_stopping']
    else:
        early_stopping = np.inf

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],
                                 weight_decay=hparams["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5)

    best_loss = np.inf
    best_epoch = 0
    best_state = {}

    for epoch in range(hparams["max_epochs"]):
        model.train()
        for data in data_loader_train:
            optimizer.zero_grad()
            data.to(hparams["device"])
            loss_train = loss_graph_classification(hparams, model, data)
            loss_train.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_val = 0
            for data in data_loader_valid:
                data.to(hparams["device"])
                loss_val += loss_graph_classification(hparams, model, data)
            loss_val /= len(data_loader_valid)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

        lr_scheduler.step()

        if epoch - best_epoch > early_stopping:
            if hparams["logging"]:
                logging.info(f"early stopping at epoch {epoch}")
            break

    if hparams["logging"]:
        logging.info('best_epoch', best_epoch)
    model.load_state_dict(best_state)
    return model.eval()


def loss_graph_classification(hparams, model, data):
    x, edge_idx = sample_graphs(data, hparams, batch_size=1)
    logits = model(x, edge_idx, data.batch)
    return F.cross_entropy(logits, data.y)
