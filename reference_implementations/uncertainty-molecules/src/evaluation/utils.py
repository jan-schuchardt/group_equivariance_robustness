import torch
import numpy as np


from src.models.standard.dimenet_pp import DimeNetPP as DropoutDimeNetPP

from torch_geometric.datasets import MD17
from src.datasets.qm7x import QM7X
from torch_geometric.loader import DataLoader

from src.evaluation.combined_dataset import CombinedDataset


def init_and_load_model(model_name, model_params, path_to_trained):
    """
    Initializes model, moves to correct device, loads trained weights and sets correct mode (train vs. eval)
    """
    if model_name == "dropout_dimenet++":
        model = DropoutDimeNetPP(**model_params)
    elif model_name == "evidential_dimenet++":
        model = EvidentialDimeNetPP(**model_params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_trained))
    
    if model_name == "dropout_dimenet++":
        model.train()
    elif model_name == "evidential_dimenet++":
        model.eval()
    
    return model

def prepare_dataset(dataset, standardize=False, num_train=1000, num_val=1000, num_test=1000, get_test_data=False, **kwargs):
    random_state = np.random.RandomState(seed=42)
    perm = torch.from_numpy(random_state.permutation(np.arange(len(dataset))))
    train_indices, val_indices, test_indices = perm[:num_train], perm[num_train:num_train+num_val], perm[num_train+num_val:]
    
    if standardize:
        mean = torch.mean(dataset.data.energy[train_indices])
        dataset.data.energy = dataset.data.energy - mean

    if get_test_data:
        test_indices = test_indices[:num_test]
        return dataset[test_indices]
    else:
        return dataset[val_indices]

def get_id_dataloader(dataset_name, dataset_params):
    if dataset_name == "MD17":
        path = '/nfs/staff-hdd/wollschl/datasets/MD17'
        dataset = MD17(path, dataset_params['molecule_name'])
        dataset = prepare_dataset(dataset, **dataset_params)
        return DataLoader(dataset, batch_size=dataset_params['val_batch_size'], shuffle=False)
        
    elif dataset_name == "QM7X":
        path = '/nfs/staff-hdd/wollschl/datasets/QM7X'
        dataset = QM7X(path, is_equilibrium=dataset_params['is_equilibrium'])
        # if dataset_params['full_dataset']:
        #     num_train = dataset_params['num_train']*10
        #     num_val = dataset_params['num_val']#*10 REMOVE COMMENT FOR FULL EVAL!!!!!
        #     dataset = prepare_dataset(dataset, standardize=dataset_params['standardize'], num_train=num_train, num_val=num_val, get_test_data=dataset_params['get_test_data'])
        # else:
        dataset = prepare_dataset(dataset, **dataset_params)
        return DataLoader(dataset, batch_size=dataset_params['val_batch_size'], shuffle=False)
    
def get_ood_dataloader(dataset_name, dataset_params):
    if dataset_name == "MD17":
        path = '/nfs/staff-hdd/wollschl/datasets/MD17'
        ood_dataset = MD17(path, dataset_params['ood_molecule_name'])
        ood_dataset = prepare_dataset(ood_dataset, **dataset_params)
        return DataLoader(ood_dataset, batch_size=dataset_params['val_batch_size'], shuffle=False)
        
    elif dataset_name == "QM7X":
        path = '/nfs/staff-hdd/wollschl/datasets/QM7X'
        ood_dataset = QM7X(path, is_equilibrium=not dataset_params['is_equilibrium'])
        ood_dataset = prepare_dataset(ood_dataset, **dataset_params)
        return DataLoader(ood_dataset, batch_size=dataset_params['val_batch_size'], shuffle=False)

        
def get_combined_dataloader(dataset_name, dataset_params):
    if dataset_name == "MD17":
        path = '../data/MD17'
        id_dataset = MD17(path, dataset_params['molecule_name'])
        id_dataset = prepare_dataset(id_dataset, **dataset_params)

        ood_dataset = MD17(path, dataset_params['ood_molecule_name'])
        ood_dataset = prepare_dataset(ood_dataset, **dataset_params)

        combined_dataset = CombinedDataset(id_dataset, ood_dataset, 'data/test')
        return DataLoader(combined_dataset, batch_size=dataset_params['val_batch_size'], shuffle=False)
        
    elif dataset_name == "QM7X":
        path = '../data/QM7X'
        ood_dataset = QM7X(path, is_equilibrium=not dataset_params['is_equilibrium'], full_dataset=False)
        ood_dataset = prepare_dataset(ood_dataset, **dataset_params)
        
        id_dataset = QM7X(path, is_equilibrium=dataset_params['is_equilibrium'], full_dataset=dataset_params['full_dataset'])
        if dataset_params['full_dataset']:
            dataset_params['num_train'] = dataset_params['num_train']*10
            dataset_params['num_val'] = dataset_params['num_val']*10
        id_dataset = prepare_dataset(id_dataset, **dataset_params)
        
        combined_dataset = CombinedDataset(id_dataset, ood_dataset, 'data/test')
        return DataLoader(combined_dataset, batch_size=dataset_params['val_batch_size'], shuffle=False)
