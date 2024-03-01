import numpy as np
import torch
from src.datasets.data_provider import get_dataloader
from src.models.standard.feature_extractors import *
from src.models.localized.feature_extractors import DimeNetPPMulti, SchNetMult, DimeNetPPMultiDropout, SchNetMultDropout
from src.models.standard.gp import load_dklgp
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from src.models.interfaces import BaseEnergyModel, BaseForceModel, ForceModel, EnergyModel
from functools import partial
from src.utils import *
from src.datasets.data_provider import get_dataloader
from os.path import join
import json
from src.models.standard.gp import Standard_GPModel, ExponentialLinear
from gpytorch.kernels import RBFKernel, MaternKernel, RQKernel, LinearKernel, ProductKernel, SpectralMixtureKernel, ScaleKernel
from tqdm import tqdm
from sklearn.cluster import KMeans

def calc_inducing_points(model, train_loader, type, n_inducing_points, per_cluster_lengthscale=True):
    print("inducing point init: ", type)
    ys = []
    embs = []
    with torch.no_grad():
        model = model.cpu()
        model = model.cuda()
        for batch in tqdm(train_loader):
            batch = batch.cuda()
            output = model(batch.cuda())
            embs.append(output.cpu())
            ys.append(batch.energy.cpu())
        embs = torch.cat(embs).cpu()
        if type == "mult-gp":
            ys = []
            per_atom_emb = {}
            max_atoms = 0
            for batch in tqdm(train_loader):
                embedding = model(batch.cuda())
                ys.append(batch.energy.cpu())
                for i in range(max(batch.batch) + 1):
                    current = embedding[batch.batch == i]
                    max_atoms = max(max_atoms, len(current))
                    for j in range(max_atoms):
                        if j in per_atom_emb:
                            per_atom_emb[j].append(current[j:j+1, :])
                        else:
                            per_atom_emb[j] = [current[j:j+1, :]]
                            
            inducing_points = []
            lengthscales = []
            num_per_cluster = n_inducing_points // max_atoms
            for k in per_atom_emb:
                embs = torch.cat(per_atom_emb[k])
                mask = np.random.choice(len(embs), size=num_per_cluster)
                inducing_points.append(embs[mask])
                if per_cluster_lengthscale:
                    try: 
                        lengthscales.append(torch.pdist(embs.cpu()).mean())
                    except:
                        lengthscales.append(torch.pdist(embs.cpu()[torch.randint(0, len(embs), (10000,))]).mean())
                        
            inducing_points = torch.cat(inducing_points)
            
            if per_cluster_lengthscale:
                # calculate lengthscale only within each cluster
                initial_lengthscale = torch.mean(torch.tensor(lengthscales))
            else:
                # calculate lengthscales also between the clusters
                initial_lengthscale = torch.pdist(inducing_points)
        elif type == "k-means" or type == "kmeans":
            data = embs.numpy()
            kmeans = KMeans(n_clusters=n_inducing_points).fit(data)
            inducing_points = torch.tensor(kmeans.cluster_centers_)
            try:
                initial_lengthscale = torch.pdist(embs.cpu()).mean()
            except:
                initial_lengthscale = torch.pdist(embs.cpu()[torch.randint(0, len(embs), (50000,))]).mean()
        elif type == "first":
            inducing_points = torch.cat(embs)[:n_inducing_points]
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        elif type == "random":
            inducing_points = torch.randn((n_inducing_points, embs.shape[1]))
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        else:
            raise NotImplementedError
        energies = torch.cat(ys)[:n_inducing_points] # energys dont make sense for kmeans!!
    return inducing_points, energies, initial_lengthscale
        
        
    inducing_points = torch.cat(embs)[:uq_params['n_inducing_points']]
    energies = torch.cat(ys)[:uq_params['n_inducing_points']]


def get_standard_model(
    encoder_name, 
    encoder_dim, 
    encoder_params, 
    uq_name, 
    uq_params, 
    train_loader, 
    batch_normalization, 
    pretrained=None,
    batch_size=None,
    ):
    fe = load_feature_extractor(encoder_name, encoder_params, pretrained)
    fe.to_encoder(output_dim=encoder_dim)
    if batch_normalization:
        fe = NormalizedModel(fe, encoder_dim)
    if uq_name == 'gp':
        model = load_dklgp(fe, uq_params, train_loader, batch_size=batch_size, embedding_dimension=encoder_dim)
    else:
        raise NotImplementedError
    return model
        
        
def load_feature_extractor(model_name, model_params, pretrained):
    model = init_model(model_name, model_params)
    if pretrained is not None:
        model.load(pretrained)
    return model


def load_fixed_fe(use_pretrained=True):
    encoder_name = "dimenet"
    encoder_params = {
        'hidden_channels': 128,
        'num_blocks': 6,
        'num_bilinear': 8,
        'num_spherical': 7,
        'num_radial': 6,
        'cutoff': 5.0,
        'envelope_exponent': 5,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_output_layers': 3,
        'out_channels': 1
    }
    if use_pretrained:
        pretrained = '/nfs/homedirs/wollschl/staff/uncertainty-molecules/models/dimenet_pretrained_U0'
    else:
        pretrained = None
    return load_feature_extractor(encoder_name, encoder_params, pretrained)


def get_fixed_standard_model(train_loader, use_pretrained=True):
    encoder_name = "dimenet"
    encoder_dim = 10
    encoder_params = {
        'hidden_channels': 128,
        'num_blocks': 6,
        'num_bilinear': 8,
        'num_spherical': 7,
        'num_radial': 6,
        'cutoff': 5.0,
        'envelope_exponent': 5,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_output_layers': 3,
        'out_channels': 1
    }
    uq_name = "gp"
    uq_params = {
        'n_inducing_points': 10,
        'num_outputs': 1
    }
    loss_fn = 'elbo' 
    beta = 1.0
    batch_normalization = False 
    if use_pretrained:
        pretrained = '/nfs/homedirs/wollschl/staff/uncertainty-molecules/models/dimenet_pretrained_U0'
    else:
        pretrained = None
    return get_standard_model(
        encoder_name,
        encoder_dim,
        encoder_params,
        uq_name,
        uq_params,
        train_loader, 
        batch_normalization,
        pretrained
    )


def init_gp(inducing_points, type, kernel_name):
    if type in ['standard-gp', 'StandardGP']:
        kernel = init_kernel(kernel_name)
        gp = Standard_GPModel(inducing_points, kernel)
    else:
        raise NotImplementedError(f'kernel {type} is not implemented yet')
    return gp

def init_kernel(kernel_name, kernel_params=None):
    if kernel_params is not None:
        if "batch_shape" in kernel_params:
            kernel_params = kernel_params.copy()
            kernel_params["batch_shape"] = torch.Size([kernel_params["batch_shape"]])
    if kernel_name == 'rbf':
        kernel = RBFKernel(**kernel_params)
    elif kernel_name == 'matern':
        kernel = MaternKernel(**kernel_params)
    elif kernel_name == 'product_explinear_rbf':
        exp_linear = ExponentialLinear(**kernel_params['linear'])
        rbf = RBFKernel(**kernel_params['rbf'])
        kernel = ProductKernel(exp_linear, rbf)
    elif kernel_name == 'product_explinear_laplace':
        pass
    elif kernel_name == 'spectral_mixture':
        kernel = SpectralMixtureKernel(**kernel_params)
    else:
        raise NotImplementedError(f"kernel {kernel_name} is not implemented")
    return kernel

def init_model(model_name, model_params):
    if model_name == "dimenet":
        return DimeNet(dimenet_params=model_params)
    elif model_name == "dimenet-mod":
        return DimeNetMod(dimenet_params=model_params)
    elif model_name in ["dimenet_pp", "dimenet++"]:
        return DimeNetPP(dimenet_pp_params=model_params)
    elif model_name in ["dimenet_pp_alt", "dimenet_pp_alternative", "dimenet++_alt", "dimenet++_alternative"]:
        return DimeNetPP_Alt(dimenet_pp_params=model_params)
    elif model_name in ["dimenet_pp_dropout", "dimenet++_dropout"]:
        return DimeNetPPDropout(dimenet_pp_params=model_params)
    elif model_name in ["dimenet_pp_dropout_mult", "dimenet++_dropout_mult"]:
        return DimeNetPPMultiDropout(dimenet_pp_params=model_params)
    elif model_name in ["dimenet_pp_mult", "dimenet++_mult"]:
        return DimeNetPPMulti(dimenet_params=model_params)
    elif model_name == "no-model":
        return NoModel()
    elif model_name == "gemnet":
        return GemNet()
    elif model_name == "schnet":
        return SchNet(params=model_params)
    elif model_name == "schnet_mult":
        return SchNetMult(params=model_params)
    elif model_name == "schnet_dropout":
        return SchNetDropout(params=model_params)
    elif model_name == "schnet_dropout_mult":
        return SchNetMultDropout(params=model_params)
    elif model_name == "painn":
        return PaiNN(params=model_params)
    elif model_name == 'spherenet':
        return SphereNet(params=model_params)
    else:
        print(f'Model {model_name} is not implemented')
        raise NotImplementedError

def wrap_prediction_model(type, model, loss_name, num_data, beta, rho_force, model_class='UQ'):
    likelihood = GaussianLikelihood()
    if loss_name is None or loss_name == 'elbo':
        loss_fn = partial(
            neg_var_elbo,
             elbo=VariationalELBO(likelihood, model.gp, num_data=num_data, beta=beta)
        )
    elif type == 'energy' and loss_name == 'mae': 
        loss_fn = torch_mae
    elif type == 'energy' and loss_name == 'torch_mae':
        loss_fn = torch_mae
    elif type == 'force' and loss_name == 'energy_mae_force_mae':
        loss_fn = partial(
            force_loss, 
            rho_force=rho_force,
            energy_loss_type="mae",
            force_loss_type="mae"
        )
    elif type == 'force' and loss_name == 'energy_elbo_force_mae':
        energy_fn = partial(
            neg_var_elbo, 
            elbo=VariationalELBO(likelihood, model.gp, num_data=num_data, beta=beta)
        )
        force_fn = torch_mae
        loss_fn = partial(
            comb_force_loss, 
            energy_fn=energy_fn, 
            force_fn=force_fn,
            rho_force=rho_force
        )
    elif type == 'force' and loss_name == 'energy_mae_force_rmse':
        loss_fn = partial(
            force_loss, 
            rho_force=rho_force,
            energy_loss_type="mae",
            force_loss_type="rmse"
        )
    
    if type == 'energy' and model_class == 'UQ':
        wrapped_model = EnergyModel(model, loss_fn, likelihood)
    elif type == 'force' and model_class == 'UQ':
        wrapped_model = ForceModel(model, loss_fn, likelihood)
    elif type == 'energy' and model_class == 'base':
        wrapped_model = BaseEnergyModel(model, loss_fn)
    elif type == 'force' and model_class == 'base':
        wrapped_model = BaseForceModel(model, loss_fn)
    else:
        raise NotImplementedError(f'type {type} is not implemented so far')
    return wrapped_model


def map_state_dict(state_dict, batch_norm):
    adj_dict = {}
    batch_norm = False
    for key in state_dict:
        if key[:13] == 'model.loss_fn':
            continue
        if key[:16] == "model.likelihood":
            continue
        if batch_norm:
            new_key = key.replace("model.model.fe.model", "fe.model.model")
        else:
            new_key = key.replace("model.model.fe.model", "fe.model")
        new_key = new_key.replace("model.model.gp", "gp")
        adj_dict[new_key] = state_dict[key]
    return adj_dict

def load_trained_gp(exp_path, model_path):
    f = open(join(exp_path, "wandb/latest-run/files/wandb-metadata.json"))
    config = json.load(f)
    params = list_to_dict(config["args"][1:])
    train_loader, val_loader, test_loader, _ = get_dataloader(
        params["dataset"], params["target"], seed=params["data_seed"], batch_size=32, normalizing=True
    )
    model = get_standard_model(
        params["encoder_name"], params["encoding_dim"], params["encoder_params"], 
        params["uq_name"], params["uq_params"], train_loader, False
    )
    # load state dict
    state_dict = torch.load(
        join(exp_path, model_path)
    )['state_dict']
    
    if "batch_normalization" in params:
        batch_norm = params["batch_norm"]
    else: 
        batch_norm = False
    
    model.load_state_dict(map_state_dict(state_dict, batch_norm))
    # load likelihood
    likelihood_dict = {}
    likelihood_dict['noise_covar.raw_noise'] = state_dict['model.likelihood.noise_covar.raw_noise']
    likelihood_dict['noise_covar.raw_noise_constraint.lower_bound'] = state_dict['model.likelihood.noise_covar.raw_noise_constraint.lower_bound']
    likelihood_dict['noise_covar.raw_noise_constraint.upper_bound'] = state_dict['model.likelihood.noise_covar.raw_noise_constraint.upper_bound']
    likelihood = GaussianLikelihood()
    likelihood.load_state_dict(likelihood_dict)
    return model, likelihood, train_loader, val_loader, test_loader
    
    
