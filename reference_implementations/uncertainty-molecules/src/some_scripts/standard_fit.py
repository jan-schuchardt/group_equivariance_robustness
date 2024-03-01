#!/usr/bin/env python3
import os
import logging
import torch
from datetime import datetime
import wandb
from src.datasets.data_provider import get_dataloader
from src.models.standard.gp import *
from sacred import Experiment
import seml
import os
from datetime import datetime
from src.datasets.data_provider import get_dataloader
from src.models.utils import load_feature_extractor


ex = Experiment()
seml.setup_logger(ex)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
                db_collection, overwrite=overwrite))

from src.models.standard.gp import GP, DKL_GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

@ex.automain
def run(model_type, n_inducing_points, useless_param):
    
    
    dataset = "QM9_dimenet"
    target = "U0"
    data_seed = 0
    encoding_dim = 1
    batch_size = 32
    debug = False
    optimizer = 'adam'
    training_iter = 50_000
    logdir = "/nfs/staff-ssd/wollschl/uncertainty-molecules/src/experiments/"

    uq_params = {
        'n_inducing_points': n_inducing_points,
        'kernel': 'RBF',
        'num_outputs': 1
    }
    local_vars = locals()
    config = {}
    for k in local_vars.keys():
        if not k.startswith('_'):
            config[k] = local_vars[k]
    n_inducing_points = uq_params['n_inducing_points']
    print(n_inducing_points)
    logging.getLogger().setLevel(logging.DEBUG)
    run_id = 0
    db_collection = 'single-batch'
    from os.path import join
    directory = join(logdir, dataset, db_collection, 
                     datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(run_id))
    logging.info(f"Directory: {directory}")
    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    train_loader, val_loader, test_loader, normalizing_constants = get_dataloader(
        dataset_name=dataset, 
        target=target,
        seed=data_seed,
        dimension=encoding_dim,
        batch_size=batch_size,
        debug=debug
    )

    diffs = []
    for batch in train_loader:
        diffs.append(batch.y.reshape(-1) - batch.x.reshape(-1))

    likelihood = GaussianLikelihood()
    fe = load_feature_extractor('no-model', {}, None)
    train_x = train_loader.dataset.inputs[range(1000)]
    train_y = train_loader.dataset.targets[range(1000)]
    X_test = val_loader.dataset.inputs
    y_test = val_loader.dataset.targets

    train_x = torch.tensor(train_x.numpy().round(2))
    train_y = torch.tensor(train_y.numpy().round(2))
    test_x = torch.tensor(X_test.reshape(-1))[::4]
    test_y = torch.tensor(y_test.reshape(-1))[::4]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    if model_type == 'exact_gp':
        model = ExactGPModel(
                    train_x=train_x, 
                    train_y=train_y, 
                    likelihood=likelihood
                ).cuda()
        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    elif model_type == 'approx_gp':
        model = load_gp_old(uq_params, fe, train_loader).cuda()
        loss_fn = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset), beta=1.0).cuda()
    else: 
        raise NotImplementedError
    
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood


    wandb.init(
        name=join(str(db_collection), str(run_id)),
        project='uncertainty-molecules',
        config={
            "model_tpye": model_type,
            "n_inducing_points": n_inducing_points
        } 
    )


    for i in range(training_iter):
        model.train()
        # Zearo gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -loss_fn(output, train_y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            with torch.no_grad():
                model.eval()
                output = model(train_x)
                mae = torch.mean(torch.abs(output.mean - train_y))
                val_output = model(test_x.cuda())
                val_mae = torch.mean(torch.abs(val_output.mean - test_y))
                print('Iter %d/%d - Loss: %.3f   Mae: %.5f   Stds: %.5f   val mae: %.3f   val stds: %.3f' % (
                    i, training_iter, loss.item(), mae.item(), output.variance.mean().item(),
                    val_mae, val_output.variance.mean().item()
                    #approx_gp.likelihood.noise.item()
                ))
                wandb.log({"loss": loss.item(), "epoch": i,
                           "mae": mae.item(), "stddevs": output.variance.mean().item(),
                           "val_mae": val_mae, "val_stddevs": val_output.variance.mean().item()})
    return {}


