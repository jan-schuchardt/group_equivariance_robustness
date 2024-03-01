import logging
import numpy as np
import seml
import torch
from sacred import Experiment as SacredExperiment

from experiment import Experiment

ex = SacredExperiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
            db_collection, overwrite=overwrite))


@ex.automain
def run(_config, conf: dict, hparams: dict):

    experiment = Experiment()
    results, dict_to_save = experiment.run(hparams)

    save_dir = conf["save_dir"]
    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    if conf["save"]:
        torch.save(dict_to_save, f'{save_dir}/{db_collection}_{run_id}')
    return results
