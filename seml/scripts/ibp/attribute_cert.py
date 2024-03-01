import numpy as np
import seml
import torch
from robust_gcn.robust_gcn import sparse_tensor
from sacred import Experiment
from scipy.sparse import csr_matrix
from tqdm import tqdm

from equivariance_robustness.interval_bound_propagation.certification import \
    certify_ged_ibp
from equivariance_robustness.attributes_zuegner.models import RobustGCNModelGED
from equivariance_robustness.attributes_zuegner.training import train
from equivariance_robustness.data import (load_node_classification_dataset,
                                          split)
from equivariance_robustness.utils import set_seed

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
        'root': '/nfs/staff-ssd/schuchaj/datasets/planetoid',
        'n_per_class': 20
    }

    training_params = {
        'n_iters': 3000,
        'method': 'Normal',
        'learning_rate': 1e-3,
        'weight_decay': 5e-4,
        'early_stopping': 50
    }

    certification_params = {
        'q': 0.01,
        'q_relative': True,
        'Q_max': 100,
        'batch_size': 8,
        'cost_add': 1,
        'cost_del': 1,
        'apply_relu': True
    }

    hidden_sizes = [32]

    seed = 0
    save_dir = '/nfs/staff-hdd/schuchaj/equivariance_certification_results/interval_bound_propagation'

train = ex.capture(train, prefix='training_params')
certify_ged = ex.capture(certify_ged_ibp, prefix='certification_params')
load_node_classification_dataset = ex.capture(load_node_classification_dataset, prefix='dataset_params')
split = ex.capture(split, prefix='dataset_params')

@ex.automain
def main(_config, seed, hidden_sizes, certification_params, save_dir):

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Load data
    data = load_node_classification_dataset(device=torch.device('cpu'), seed=seed)

    edge_idx = data.edge_index.numpy()
    X = data.x.numpy()
    y = data.y.numpy().astype('int8')
    A = csr_matrix((np.ones(len(edge_idx[0])), (edge_idx[0], edge_idx[1])))
    X = csr_matrix(X)
    K = y.max()+1
    N,D = X.shape
    X_t = sparse_tensor(X).to(device)
    y_t = torch.tensor(y.astype("int64"), device=device)

    # Split data
    idx_train, idx_val, idx_test = split(y, n_per_class=20)

    # Create model
    model = RobustGCNModelGED(A, [D]+hidden_sizes+[K]).cuda()

    # Train model
    q = certification_params['q']
    if certification_params['q_relative']:
        q = q * D
    q = int(q)

    train(gcn_model=model, X=X_t, y=y_t, idx_train=idx_train, idx_val=idx_val)
    model.eval()
    pred = model.predict(X_t.to_dense(), np.arange(N)).detach().cpu().numpy()

    results_dict = {
        'targets': y,
        'pred': pred,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
        'test_accuracy': (pred[idx_test] == y[idx_test]).mean(),
        'config': _config
    }

    # Certify model

    certified = []
    max_rad = 0


    for Q in tqdm(range(1, certification_params['Q_max'])):
        certified_at_Q = certify_ged_ibp(gcn_model=model, attrs=X.astype("float32"), q=q, 
                                Q=Q, progress=False)[0]

        certified.append(certified_at_Q)

        if np.sum(certified_at_Q) == 0:
            break
        else:
            max_rad += 1

    certified = np.vstack(certified)

    results_dict.update({
        'certified': certified,
        'max_rad': max_rad
    })

    results_dict['certified'] = certified
    results_dict['max_rad'] = max_rad


    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    save_file = f'{save_dir}/{db_collection}_{run_id}'

    torch.save(results_dict,
               save_file)

    return {
        'test_accuracy': results_dict['test_accuracy'],
        'max_rad': max_rad,
        'save_file': save_file
    }
