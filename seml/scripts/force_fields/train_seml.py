#!/usr/bin/env python3
import seml
import wandb
import os
import logging
import torch
import pickle
from datetime import datetime
from sacred import Experiment
from os.path import join

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger  # newline 1

from equivariance_robustness.force_fields.models import CenterSmoothedForceModel

from src.models.utils import load_feature_extractor, wrap_prediction_model
from src.datasets.data_provider import get_dataloader
from src.training.trainer import ModelWrapper
from src.training.optimizer import Optimizer
from src.models.standard.gp import *

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
                db_collection, overwrite=overwrite))


@ex.automain
def run(_config, model_name, dataset, data_seed, num_epochs, optimizer_params,
        batch_size, target, loss, optimizer, scheduler, ema_decay, patience, save_ema,
        scheduler_params, logdir, model_params={}, model_seed=1, pretrained=None, debug=False, normalize=True, num_datapoints=None,
        beta=1.0, prediction_type='energy', rho_force=None, val_batch_size=None, dataset_params={}, use_min_metric=True,
        smoothing_params={}):
    torch.manual_seed(model_seed)
    local_vars = locals()
    config = {}
    for k in local_vars.keys():
        if not k.startswith('_'):
            config[k] = local_vars[k]
    logging.getLogger().setLevel(logging.DEBUG)
    run_id = _config['overwrite']
    db_collection = _config['db_collection']
    directory = join(
        logdir, 
        dataset, 
        db_collection, 
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(run_id)
        )
    logging.info(f"Directory: {directory}")
    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)


    train_loader, val_loader, _, normalizing_constants = get_dataloader(
        dataset_name=dataset, 
        target=target,
        seed=data_seed,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        debug=debug,
        normalizing=normalize,
        num_datapoints=num_datapoints,
        **dataset_params
    )

    #ood_set = valset
    if pretrained is not None:
        pretrained_name = pretrained + str(target)
    else:
        pretrained_name = None
    model = load_feature_extractor(model_name, model_params, pretrained_name)

    # either energy prediction or force prediction
    model = wrap_prediction_model(
        prediction_type, model, loss, 
        len(train_loader.dataset), beta, rho_force=rho_force,
        model_class='base'
    )

    model = CenterSmoothedForceModel(model, **smoothing_params)

    optim = Optimizer(
        optimizer, 
        optimizer_params,
        scheduler, 
        scheduler_params
        )

    lightning_model = ModelWrapper(
        model, 
        optim, 
        ema_decay=ema_decay,
        train_loader=train_loader, 
        val_loader=val_loader,
        batch_size=batch_size, #
        save_ema=save_ema
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss', 
        save_top_k=1
        )
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode="min")
    wandb.init(
        dir=os.path.abspath(directory), 
        name=join(str(db_collection), str(run_id)),
        project='uncertainty-molecules'
    )
    if use_min_metric:
        wandb.define_metric('val_energy_mae', summary='min')
        wandb.define_metric('val_force_mae', summary='min')
        wandb.define_metric('val_loss', summary='min')
        wandb.define_metric('train_energy_mae', summary='min')
        wandb.define_metric('train_force_mae', summary='min')
        wandb.define_metric('train_loss', summary='min')
    wandb_logger = WandbLogger(
        save_dir=os.path.abspath(directory), 
        name=join(str(db_collection), str(run_id)),
        project='uncertainty-molecules'
    )
    with open(wandb_logger.save_dir+'/config.pkl', 'wb') as f:
        print('saving config at: ', wandb_logger.save_dir)
        pickle.dump(config, f)
    with open(wandb_logger.save_dir+'/mean_and_td.pkl', 'wb') as f:
        print('saving normalizing constants at: ', wandb_logger.save_dir)
        pickle.dump(normalizing_constants, f)

    wandb_logger.log_hyperparams(config)
    trainer = pl.Trainer(callbacks=[lr_monitor, checkpoint_callback, progress_bar, early_stopping], 
                         log_every_n_steps=50, gpus=1, max_epochs=num_epochs, logger=wandb_logger, precision=32)
    trainer.test(lightning_model)
    trainer.fit(lightning_model)

    results = {}
    for key in trainer.logged_metrics:
        if isinstance(trainer.logged_metrics[key], torch.Tensor):
            results[key] = trainer.logged_metrics[key].item()
        else:
            results[key] = trainer.logged_metrics[key]

    weight_directory = join(
        logdir,
        dataset,
        db_collection,
        'weights'
    )

    if not os.path.exists(weight_directory):
        os.makedirs(weight_directory)

    weight_file = f'{weight_directory}/{run_id}'

    torch.save(lightning_model.model.state_dict(),
               weight_file)

    results['weight_file'] = weight_file

    print(results)

    return results
