from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, APPNP, GINConv, JumpingKnowledge, global_mean_pool
from torch_sparse import SparseTensor, set_diag, spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from collections import OrderedDict
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp

import torch
import random
import numpy as np


def create_model(hparams, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    arch = hparams['arch']
    if arch == "GAT":
        model = GAT(hparams)
    elif arch == "GCN":
        model = GCN(hparams)
    elif arch == "APPNP":
        model = APPNPNet(hparams)
    elif arch == "SimpleGCN":
        model = SimpleGCN(hparams)
    else:
        raise Exception("Not implemented")
    return model.to(hparams["device"])


class GAT(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.conv1 = GATConv(hparams['in_channels'],
                             hparams['hidden_channels'],
                             heads=hparams['k_heads'],
                             edge_dim=1,
                             dropout=hparams['p_dropout'])
        self.conv2 = GATConv(hparams['k_heads']*hparams['hidden_channels'],
                             hparams['out_channels'],
                             edge_dim=1,
                             dropout=hparams['p_dropout'])
        self.p_dropout = hparams['p_dropout']

    def forward(self, x, edge_idx, edge_attr=None):
        hidden = F.elu(self.conv1(x, edge_idx, edge_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        return self.conv2(hidden, edge_idx, edge_attr)


class GCN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.conv1 = GCNConv(hparams['in_channels'],
                             hparams['hidden_channels'])
        self.conv2 = GCNConv(hparams['hidden_channels'],
                             hparams['out_channels'])

        self.p_dropout = hparams['p_dropout']

    def forward(self, x, edge_idx, edge_attr=None):
        hidden = F.relu(self.conv1(x, edge_idx, edge_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        return self.conv2(hidden, edge_idx, edge_attr)


class APPNPNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.lin1 = nn.Linear(hparams["in_channels"],
                              hparams["hidden_channels"],
                              bias=False)
        self.lin2 = nn.Linear(hparams["hidden_channels"],
                              hparams["out_channels"],
                              bias=False)
        # k_hops=10, appnp_alpha=0.15
        self.prop = APPNP(hparams["k_hops"], hparams["appnp_alpha"])
        self.p_dropout = hparams["p_dropout"]

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_idx):
        hidden = F.relu(self.lin1(x))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.lin2(hidden)
        return self.prop(hidden, edge_idx)


class SimpleGCN(nn.Module):
    # https://github.com/RobustGraph/RoboGraph
    def __init__(self, hparams):
        super().__init__()
        self.conv = GCNConv(hparams['in_channels'],
                            hparams['hidden_channels'])  # init with 64
        self.lin = nn.Linear(hparams["hidden_channels"],
                             hparams["out_channels"])

    def forward(self, x, edge_idx, batch):
        hidden = F.relu(self.conv(x, edge_idx))
        hidden = global_mean_pool(hidden, batch)
        return self.lin(hidden)
