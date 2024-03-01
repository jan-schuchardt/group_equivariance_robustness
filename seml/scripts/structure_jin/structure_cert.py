import os.path as osp

import numpy as np
import seml
import torch
from robograph.model.gnn import GC_NET
from sacred import Experiment
from tqdm import tqdm
from tqdm import tqdm

from equivariance_robustness.data import load_graph_classification_dataset
from equivariance_robustness.structure_jin.training import train_classifier
from equivariance_robustness.utils import set_seed
from equivariance_robustness.structure_jin.certification import certify

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
            )

    dataset_params = {
        'name': 'Citeseer',
        'root': '/nfs/staff-ssd/schuchaj/datasets/tudataset',
    }

    training_params = {
        'num_epochs': 200,
        'batch_size': 20
    }

    certification_params = {
        'cost_add': 1,
        'cost_del': 1,
        'local_strength': 3,
        'max_global_budget': 100,
        'budget_steps': 1,
        'dual_iterations': 100,
        'max_degree': None
    }

    dim_hidden = 64
    dropout = 0.0

    seed = 0
    save_dir = '/nfs/staff-hdd/schuchaj/equivariance_certification_results/structure_jin'

load_graph_classification_dataset = ex.capture(load_graph_classification_dataset, prefix='dataset_params')
train_classifier = ex.capture(train_classifier, prefix='training_params')
certify = ex.capture(certify, prefix='certification_params')

@ex.automain
def main(_config, seed, dim_hidden, dropout, certification_params, save_dir):

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Load data

    train_dataset, val_dataset, test_dataset = load_graph_classification_dataset(
                                                    seed=seed, return_dataloader=False)

    model = GC_NET(hidden=dim_hidden,
              n_features=train_dataset.num_features,
              n_classes=train_dataset.num_classes,
              act='relu',
              pool='avg',
              dropout=dropout).to(device)

    train_acc_history, val_acc_history, test_acc = train_classifier(
                    model=model, train_dataset=train_dataset,
                    val_dataset=val_dataset, test_dataset=test_dataset)


    model.eval()

    targets = [int(graph.y) for graph in test_dataset]

    certified_masks = []
    predictions = None

    max_rad=0
    budget_steps = certification_params['budget_steps']

    for global_budget in tqdm(range(1, certification_params['max_global_budget']+1)[::budget_steps]):
        pred, certified_at_Q = certify(
            model=model, dataset=test_dataset, global_budget=global_budget
        )

        if predictions is None:
            predictions = pred

        certified_masks.append(certified_at_Q)

        if not np.any(certified_at_Q):
            break
        else:
            max_rad += 1

    certified_masks = np.vstack(certified_masks)

    results_dict = {
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'test_acc': test_acc,
        'targets': targets,
        'predictions': predictions,
        'certified_masks': certified_masks,
        'max_rad': max_rad
    }


    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    save_file = f'{save_dir}/{db_collection}_{run_id}'

    torch.save(results_dict,
               save_file)

    return {
        'test_accuracy': results_dict['test_acc'],
        'max_rad': max_rad,
        'save_file': save_file
    }
