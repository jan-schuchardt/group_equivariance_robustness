import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch.utils.data._utils.collate import default_collate


def mycollate(data):
    print('somestuff')
    raise NotImplementedError
    return 0

def main():
    dataset = QM9("/nfs/staff-hdd/wollschl/datasets/QM9")
    loader = DataLoader(dataset, batch_size=16)
    for batch in loader:
        print(batch)


if __name__ == "__main__":
    main()