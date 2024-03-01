from abc import abstractmethod
import torch.nn as nn
import torch
from typing import Tuple
import abc
from torch.autograd import grad
from src.utils import torch_mae


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def to_encoder(self):
        raise NotImplementedError

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False
    
    def unfreeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = True
            elif name in param_list:
                param.requires_grad = True
    
    def load(self, file):
        state_dict = torch.load(file)
        if 'ema_state_dict' in state_dict:
            print('loading ema state dict')
            state_dict = map_state_dict(state_dict['ema_state_dict'])
        elif 'state_dict' in state_dict:
            state_dict = map_state_dict(state_dict['state_dict'])
        print(self.load_state_dict(state_dict))

def map_state_dict(sd):
    new_dict = {}
    for k in sd:
        if k[:16] == "model.model.out_module":
            new_dict[k[6:]] = sd[k]
        else:
            new_dict[k[12:]] = sd[k]
    return new_dict


class UQModel(nn.Module):
    def __init__(self):
        super().__init__()

    


class LightningInterface(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError
    
class ForceModel(LightningInterface):
    def __init__(self, model, loss_fn, likelihood):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.likelihood = likelihood
        
    def forward(self, batch):
        batch.pos.requires_grad = True
        torch.set_grad_enabled(True)
        energy = self.model(batch)
        forces = -grad(
                    outputs=energy.rsample().sum(), #.mean.sum(), 
                    inputs=batch.pos, 
                    create_graph=True,
                    retain_graph=True
                )[0]
        batch.pos.requires_grad = False
        return energy, forces
    
    def sample_multiple(self, batch, num_samples):
        batch.pos.requires_grad = True
        torch.set_grad_enabled(True)
        energy = self.model(batch)
        energy_samples = energy.rsample(torch.Size([num_samples]))
        forces_samples = []
        for sample in energy_samples:
            forces_samples.append(-grad(
                outputs=sample.sum(),
                inputs=batch.pos,
                create_graph=False,
                retain_graph=True #do we need to retain the graph here?
            )[0])
        batch.pos.requires_grad = False
        forces_samples = torch.stack(forces_samples)
        return energy_samples, forces_samples


    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        energy, forces = self.forward(batch)
        loss = self.loss_fn(energy, forces, batch.energy, batch.force)
        energy_mae = torch_mae(energy.mean, batch.energy)
        energy_likelihood = - self.likelihood.expected_log_prob(batch.energy, energy).mean().detach()
        energy_stddevs = energy.stddev.cpu()
        force_mae = torch_mae(forces, batch.force)
        
        return loss, {
            'energy_mae': energy_mae, 
            'energy_likelihood': energy_likelihood, 
            'energy_stddevs': energy_stddevs.mean(), 
            'force_mae': force_mae
        }     
    
class EnergyModel(LightningInterface):
    def __init__(self, model, loss_fn, likelihood):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.likelihood = likelihood
    
    def forward(self, batch):
        energy = self.model(batch)
        return energy
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        energy = self.forward(batch)
        if batch.y is not None:
            y = batch
        elif batch.energy is not None:
            y = batch.energy
        loss = self.loss_fn(energy, y)
        energy_mae = torch_mae(energy.mean, y)
        energy_likelihood = - self.likelihood.expected_log_prob(y, energy).mean().detach()
        energy_stddevs = energy.stddev.cpu()
        
        return loss, {
            'energy_mae': energy_mae, 
            'energy_likelihood': energy_likelihood, 
            'energy_stddevs': energy_stddevs.mean()
        }

class BaseEnergyModel(LightningInterface):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch):
        return self.model(batch)
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        energy = self.forward(batch).reshape(-1)
        loss = self.loss_fn(energy, batch.y)
        energy_mae = torch_mae(energy, batch.y)
        return loss, {'energy_mae': energy_mae}

class BaseForceModel(LightningInterface):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch):
        batch.pos.requires_grad = True
        torch.set_grad_enabled(True)
        energy = self.model(batch)
        forces = -grad(
                    outputs=energy.sum(), 
                    inputs=batch.pos, 
                    create_graph=True,
                    retain_graph=True
                )[0]
        batch.pos.requires_grad = False
        return energy, forces
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        energy, forces = self.forward(batch)
        energy = energy.reshape(-1)
        loss = self.loss_fn(energy, forces, batch.energy, batch.force)
        energy_mae = torch_mae(energy, batch.energy)
        force_mae = torch_mae(forces, batch.force)
        
        return loss, {
            'energy_mae': energy_mae, 
            'force_mae': force_mae
        }
