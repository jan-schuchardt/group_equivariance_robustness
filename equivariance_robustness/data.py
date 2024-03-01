from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import numpy as np


def load_node_classification_dataset(name, root, seed, device="cuda"):
    """
      - name in ["Cora", "Pubmed", "Citeseer"]
      - preprocessings: to undirected, extract lcc, 
                        remove self-loops, binarize features
    """
    transform = T.Compose([T.ToUndirected(),
                           T.LargestConnectedComponents()])
    dataset = Planetoid(root=root,
                        name=name,
                        split="random", # will define own split below
                        transform=transform)
    data = dataset[0]

    # remove self-loops
    edge_idx = data.edge_index
    data.edge_index = edge_idx[:, edge_idx[0] != edge_idx[1]]

    # binarize
    data.x[data.x > 0] = 1

    # custom split
    idx_train, idx_valid, idx_test = split(data.y.numpy(), seed=seed)

    data.train_mask.fill_(False)
    data.train_mask[idx_train] = True

    data.val_mask.fill_(False)
    data.val_mask[idx_valid] = True

    data.test_mask.fill_(False)
    data.test_mask[idx_test] = True

    return data.to(device)


def load_graph_classification_dataset(name, root, seed, return_dataloader=True):
    """
      - name in ["ENZYMES"]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = TUDataset(root=root, name=name, use_node_attr=True)
    dataset = dataset.shuffle()
    train_size = len(dataset) // 10 * 3
    val_size = len(dataset) // 10 * 2
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size: train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    if return_dataloader:
        # prepare dataloader
        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        return train_loader, val_loader, test_loader

    else:
        return train_dataset, val_dataset, test_dataset

def split(labels, n_per_class=20, seed=0):
    """
      See https://github.com/abojchevski/sparse_smoothing/blob/master/sparse_smoothing/utils.py
    """
    np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for l in range(nc):
        perm = np.random.permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)),
                              np.concatenate((split_train, split_val)))

    return split_train, split_val, split_test