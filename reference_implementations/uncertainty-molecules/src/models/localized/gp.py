from pyexpat import model
import torch
import torch.nn as nn
from torch.autograd import grad 
from typing import Tuple
from sklearn import cluster
from torch_geometric.data import Batch
import gpytorch
from gpytorch.models import ApproximateGP
from src.models.interfaces import BaseModel, LightningInterface
from src.metrics.calibration_scores import calibration_regression
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from torch_geometric.loader import DataLoader
from src.utils import torch_mae
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from torch_scatter import scatter
from src.models.localized.variational_strategy import FixedIndPoint_VariationalStrategy

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)


class SingleGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SingleGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class Fixed_IndPoint_GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, covar_module, device=torch.device("cuda")):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0) + 1)
        variational_strategy = FixedIndPoint_VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True, device=device
        )
        super(Fixed_IndPoint_GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x, batch=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultiGPModel(torch.nn.Module):
    def __init__(self, gp, reduce_mean="sum", reduce_covar="sum", full_covar=False, **kwargs):
        super().__init__()
        self.gp = gp
        self.reduce_mean = reduce_mean
        self.reduce_covar = reduce_covar
        self.full_covar = full_covar
        self.full_operator = None
        self.batch_max = 0
        self.batch_length = 0
        if full_covar:
            print('\n\n\n\n')
            print('USING FULL COVAR')
            print('\n\n\n\n')
        
    def forward(self, embeddings, batch=None):
        pred = self.gp(embeddings)
        if batch is not None: #if batch is none we are in the evaluation mode for the forces
            if self.full_covar:
                combined_mean, combined_covar = self.aggregate_covar(pred, batch)
            else:
                mean, covar = pred.mean, pred.covariance_matrix
                combined_mean = scatter(mean, batch, 0, reduce=self.reduce_mean)
                combined_covar = torch.eye(max(batch) + 1).to(covar) * scatter(covar.diag(), batch, 0, reduce=self.reduce_covar)
            pred = MultivariateNormal(combined_mean, combined_covar)
        return pred
        
    def aggregate_covar(self, energy_dist, batch):
        if self.batch_max == batch.max() and self.batch_length == len(batch) and self.full_operator is not None:
            operator = self.full_operator
        else:
            self.batch_max = batch.max()
            self.batch_length = len(batch)
            operator = torch.zeros(batch.max() + 1, len(batch))
            for i in range(batch.max() + 1):
                operator[i, batch == i] += 1
            operator = operator.to(energy_dist.loc)
            self.full_operator = operator
        mean = operator @ energy_dist.loc
        covar = operator @ energy_dist.covariance_matrix @ operator.transpose(1, 0)
        return mean, covar