#!/usr/bin/env python3
import logging
import os
import pickle
from datetime import datetime
from os.path import join

import numpy as np
import seml
import torch
from sacred import Experiment
from src.datasets.data_provider import get_dataloader
from src.models.standard.gp import *
from src.models.utils import load_feature_extractor, wrap_prediction_model
from tqdm import tqdm

from equivariance_robustness.force_fields.models import \
    CenterSmoothedForceModel
from pointcloud_invariance_smoothing.utils import dict_to_dot

ex = Experiment()
seml.setup_logger(ex)


@ex.capture(prefix='train_loading')
def get_trained_weights(collection, exp_id, restrictions):

    if exp_id is None and restrictions is None:
        raise ValueError('You must provide either an exp-id or a restriction dict')
    if collection is None:
        raise ValueError('You must a collection to load trained model from')

    mongodb_config = seml.database.get_mongodb_config()
    coll = seml.database.get_collection(collection, mongodb_config)

    if exp_id is not None:
        weight_file = coll.find_one({'_id': exp_id}, ['result'])['result']['weight_file']
    else:
        coll_filter = restrictions.copy()
        coll_filter = {'config.' + k: v for k, v in dict_to_dot(coll_filter)}

        exps = list(coll.find(coll_filter, ['result']))
        if len(exps) == 0:
            raise ValueError("Find yielded no results.")
        elif len(exps) > 1:
            raise ValueError(f"Find yielded more than one result: {exps}")
        else:
            weight_file = exps[0]['result']['weight_file']

    return torch.load(weight_file)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
                db_collection, overwrite=overwrite))


@ex.automain
def run(_config, model_name, dataset, data_seed,
        target, loss, rho_force, cert_dir, model_params={}, model_seed=1, normalize=True,
        prediction_type='force', dataset_params={}, smoothing_params={},
        n_budget_steps=1000, n_molecules=1000):
    torch.manual_seed(model_seed)
    local_vars = locals()
    config = {}
    for k in local_vars.keys():
        if not k.startswith('_'):
            config[k] = local_vars[k]
    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    _, _, test_loader, _ = get_dataloader(
        dataset_name=dataset,
        target=target,
        seed=data_seed,
        batch_size=1,
        val_batch_size=1,
        debug=False,
        normalizing=normalize,
        **dataset_params
    )

    model = load_feature_extractor(model_name, model_params, None)

    # either energy prediction or force prediction
    model = wrap_prediction_model(
        prediction_type, model, loss, 
        len(test_loader.dataset), beta=1,
        rho_force=rho_force,
        model_class='base'
    )

    model = CenterSmoothedForceModel(model, **smoothing_params).cuda()

    state_dict = get_trained_weights()
    model.load_state_dict(state_dict)

    model.eval()

    pred_errors = []
    pred_errors_mae = []
    abstains = []
    max_rad = 0
    budgets = []
    certified_output_distances = []

    for i, molecule in tqdm(enumerate(test_loader), total=n_molecules):
        if i == n_molecules:
            break

        center, radius = model.pred_center(molecule)
        pred_errors.append(float(model.d_out(center, molecule.force[np.newaxis])[0].detach()))
        pred_errors_mae.append(float((center - molecule.force).abs().mean()))
        abstains.append(bool(model.abstain(molecule, center, radius)))
        max_rad = model.calc_max_rad()
        budgets.append(torch.linspace(0, max_rad, n_budget_steps))
        certified_output_distances.append(model.certify(molecule, center, budgets[-1]).detach())

    save_dict = {
        'config': _config,
        'pred_errors': pred_errors,
        'pred_erros_mae': pred_errors_mae,
        'abstains': abstains,
        'max_rad': max_rad,
        'budgets': budgets,
        'certified_output_distances': certified_output_distances
    }

    save_dir = join(
        cert_dir,
        dataset,
        db_collection)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = f'{save_dir}/{run_id}'

    torch.save(save_dict, save_file)

    results = {
        'save_file': save_file,
        'average_pred_error': np.mean(save_dict['pred_errors'])
    }

    return results
