import numpy as np
import torch
from torch_geometric.data import Dataset

class CombinedDataset(Dataset):
    """
    Dataset of ood data + id data.
    """

    def __init__(self, id_data, ood_data, root):
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

        self.id_data = id_data
        self.ood_data = ood_data
        self.id_len = len(self.id_data)  # type: ignore
        self.ood_len = len(self.ood_data)  # type: ignore

    def len(self):
        return self.id_len + self.ood_len

    def get(self, idx):
        if idx < self.id_len:
            return self.id_data[idx], 1 # orginally 1 
        return self.ood_data[idx - self.id_len], 0 # originally 0