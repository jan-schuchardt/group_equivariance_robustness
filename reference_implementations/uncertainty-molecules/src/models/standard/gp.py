from pyexpat import model
from turtle import forward
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

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

class ExponentialLinear(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    # is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, length_prior=None, base_args=None, **kwargs):
        super().__init__(**kwargs)
        
        self.linear_kernel = gpytorch.kernels.LinearKernel()
        self.linear_kernel.requires_grad_(False)
    # this is the kernel function
    def forward(self, x1, x2, **params):
        linear_val = 1/1_000_000 * self.linear_kernel(x1, x2, **params)
        
        if not isinstance(linear_val, torch.Tensor):
            linear_val = linear_val.evaluate()
        
        return torch.exp(linear_val)# / e
    
class Standard_GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, covar_module):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(Standard_GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x, batch=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def load_gp_old(gp_params, feature_extractor, train_loader, batch_size, embedding_dimension):
    initial_inducing_points, initial_lengthscale = initial_values(
        train_loader.dataset, feature_extractor, gp_params['n_inducing_points']
        )
    print(initial_lengthscale)
    gp = GP(
        initial_inducing_points=initial_inducing_points, 
        batch_size=batch_size,
        initial_lengthscale=initial_lengthscale,
        embedding_dimension=embedding_dimension,
        **gp_params
        )
    return gp

def load_dklgp(fe, gp_params, trainset, batch_size, embedding_dimension):
    gp = load_gp_old(gp_params, fe, trainset, batch_size, embedding_dimension)
    return DKL_GP(fe, gp)
    

def initial_values(train_dataset, feature_extractor, n_inducing_points):
    #idxs = torch.randperm(len(dataset))[:1000]
    #loader = DataLoader(dataset.index_select(idxs), batch_size=100, shuffle=False, num_workers=1)
    f_X_samples = []

    counter = 0
    with torch.no_grad():
        for batch in DataLoader(train_dataset, batch_size=128):
            if counter >= 100: #maybe increase if we want to use batch norm
                break
            if torch.cuda.is_available():
                batch = batch.cuda()
                feature_extractor = feature_extractor.cuda()

            if isinstance(batch, Batch):
                f_X_samples.append(feature_extractor(batch).cpu())
            else:
                f_X_samples.append(feature_extractor(batch).cpu())
            counter += 1

    f_X_samples = torch.cat(f_X_samples)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    #if torch.cuda.is_available():
    #    f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()

class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        kernel="RBF",
        batch_size=None,
        embedding_dimension=10,
        mean_layer_type='linear',
        **kwargs
    ):
        n_inducing_points = initial_inducing_points.shape[0]

        #if batch_size is not None and batch_size > 1:
        #    batch_shape = torch.Size([batch_size])
        #else:
        batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )

        super().__init__(variational_strategy)

        kwargs = {
            "batch_shape": batch_shape,
            "eps": 1e-10
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        if mean_layer_type == 'linear':
            self.mean_module = LinearMean(embedding_dimension, batch_shape=batch_shape)
        elif mean_layer_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_layer_type == 'zero':
            self.mean_module = ZeroMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)
        from gpytorch.means import ConstantMeanGrad
        #self.mean_module = ConstantMeanGrad(batch_shape=batch_shape)
        #self.covar_module = self.covar_module.requires_grad_(False)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param


class DKL_GP(LightningInterface, gpytorch.Module):
    # types of all
    def __init__(self, feature_extractor: BaseModel, gp: GP) -> None:
        super().__init__()
        self.fe = feature_extractor
        self.gp = gp

    def forward(self, x):
        features = self.fe(x)
        output = self.gp(features)
        return output


class FeaturewiseGP(gpytorch.Module):
    def __init__(self, covar_module, num_dim, inducing_points, grid=None):
        super().__init__()
        self.gp = GaussianProcessLayer(covar_module=covar_module, num_dim=num_dim, inducing_points=inducing_points)
        if grid is not None:
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid[0], grid[1])
        else:
            self.scale_to_bounds = torch.nn.Identity()
        self.mixing = torch.nn.Parameter(torch.randn(256, 1))
        
    def forward(self, x, batch=None):
        # assuming x to be N,D
        x = x.transpose(-1, -2).unsqueeze(-1)
        multitask_normal = self.gp(x)
        indep_normal = multitask_normal.to_data_independent_dist()
        mean = (indep_normal.loc @ self.mixing).squeeze(-1)
        covar = (indep_normal.covariance_matrix @ self.mixing).squeeze(-1) @ self.mixing
        return gpytorch.distributions.MultivariateNormal(mean, torch.eye(len(mean), device=mean.device) * covar)
    
    
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, covar_module, num_dim, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0), batch_shape=torch.Size([num_dim])
        )

        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points=inducing_points, 
                variational_distribution=variational_distribution,
                learn_inducing_locations=True
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = covar_module
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)