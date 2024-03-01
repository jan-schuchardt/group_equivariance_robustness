import re
from turtle import forward
import yaml
from statistics import mean
from tkinter.messagebox import NO
import torch
from torch.nn import Identity, Linear, Sequential
from typing import Tuple
from torch_geometric.nn import DimeNet as DimeNet_Geom
from src.models.standard.dimenet_pp import DimeNetPP as DimeNetPP_Geom
from src.models.standard.dimenet_pp_dropout import  DimeNetPPDropout as DimenetDropoutBase
from src.models.standard.dimenet_pp import DimeNetPP_alternative as DimeNetPP_Alternative
from src.models.standard.schnet import SchNet as SchnetBase
from src.models.standard.schnet_dropout import SchNetDropout as SchNetDropoutBase
from src.models.standard.spherenet import SphereNet as SphereNetBase
from src.models.interfaces import BaseModel, LightningInterface
from src.utils import torch_mae as mae_loss, force_loss
from gemnet.model.gemnet import GemNet as GemNetBase
from src.utils import force_loss
from torch_geometric.nn import SchNet as SchNet_Geom
from src.models.standard.painn import PaiNN as PaiNN_Base
from schnetpack.atomistic import Atomwise
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff
from os.path import join
class GemNet(BaseModel):
    def __init__(self):
        super().__init__()
        with open("gemnet_config.yaml", "r") as stream:
            try:
                args = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.model = GemNetBase(**args)
        self.predict_forces = True

    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        energy, forces = self.model((z, pos, b))
        if self.predict_forces:
            return energy, forces
        else:
            return energy


    def to_encoder(self, output_dim=None):
        self.model.direct_forces = True # make sure we dont have double backprop
        self.predict_forces = False
        for output_block in self.model.out_blocks:
            if output_dim is None or output_dim == 128:
                output_block.out_energy = Identity()
            elif output_dim == 1:
                continue
            else:
                output_block.out_energy = Linear(128, output_dim)
        self.is_encoder = True

class DimeNet(LightningInterface, BaseModel):
    def __init__(self, dimenet_params: dict):
        super(DimeNet, self).__init__()
        self.model = DimeNet_Geom(**dimenet_params)
        self.is_encoder = False

    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        preds = self.model(batch.z, batch.pos, batch.batch)
        mae = mae_loss(preds, batch.y)
        return mae, {}

    def to_encoder(self, output_dim=None):
        for output_block in self.model.output_blocks:
            if output_dim is None or output_dim == 128:
                output_block.lin = Identity()
            elif output_dim == 1:
                continue
            else:
                output_block.lin = Linear(128, output_dim)
        self.is_encoder = True
    
    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                if self.is_encoder:
                    if name in re.findall('output_blocks\..\.lin\..*', name):
                        print('skipping freezing of parameter ', name)
                        continue
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False

class DimeNetMod(LightningInterface, BaseModel):
    def __init__(self, dimenet_params: dict):
        super(DimeNetMod, self).__init__()
        self.model = DimeNet_Geom(**dimenet_params)
        self.is_encoder = False
        
        for output_block in self.model.output_blocks:
            output_block.lin = Sequential(
                Linear(128, 10),
                Linear(10, 1)
            )
    
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        preds = self.forward(batch)
        mae = mae_loss(preds.reshape(-1), batch.y)
        return mae, {}

    def to_encoder(self, output_dim=None):
        for output_block in self.model.output_blocks:
            if output_dim is None or output_dim == 128:
                output_block.lin = Identity()
            elif output_dim == 1:
                continue
            elif output_dim == 10:
                output_block.lin[1] = Identity()
            else:
                raise NotImplementedError('for arbitrary output dimension use standard dimenet')
        self.is_encoder = True
    
    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                if self.is_encoder:
                    if name in re.findall('output_blocks\..\.lin\..*', name):
                        print('skipping freezing of parameter ', name)
                        continue
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False

class DimeNetPP(LightningInterface, BaseModel):
    def __init__(self, dimenet_pp_params: dict):
        super(DimeNetPP, self).__init__()
        self.model = DimeNetPP_Geom(**dimenet_pp_params)
        self.is_encoder = False

    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)

    def execution_step(self, batch: torch.Tensor, rho_force: float) -> Tuple[torch.Tensor, dict]:
        preds = self.model(batch.z, batch.pos, batch.batch)
        if self.model.predict_forces:
            energy, forces = preds
            loss = force_loss(energy.reshape(-1), forces, batch.energy, batch.force, rho_force=rho_force)
        else:
            if batch.y is not None:
                y = batch.y
            elif batch.energy is not None:
                y = batch.energy
            else:
                raise NotImplementedError
            energy = preds
            loss = mae_loss(energy.reshape(-1), y)
        return loss, {}

    def to_encoder(self, output_dim=None):
        for output_block in self.model.output_blocks:
            if output_dim is None or output_dim == 256:
                output_block.lin = Identity()
                output_block.to_encoder()
            elif output_dim == 1:
                continue
            else:
                output_block.lin = Linear(256, output_dim)
        self.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                if self.is_encoder:
                    if name in re.findall('output_blocks\..\.lin\..*', name):
                        print('skipping freezing of parameter ', name)
                        continue
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False
                
class DimeNetPPDropout(LightningInterface, BaseModel):
    def __init__(self, dimenet_pp_params: dict):
        super(DimeNetPPDropout, self).__init__()
        self.model = DimenetDropoutBase(**dimenet_pp_params)
        self.is_encoder = False

    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)

    def execution_step(self, batch: torch.Tensor, rho_force: float) -> Tuple[torch.Tensor, dict]:
        preds = self.model(batch.z, batch.pos, batch.batch)
        if self.model.predict_forces:
            energy, forces = preds
            loss = force_loss(energy.reshape(-1), forces, batch.energy, batch.force, rho_force=rho_force)
        else:
            if batch.y is not None:
                y = batch.y
            elif batch.energy is not None:
                y = batch.energy
            else:
                raise NotImplementedError
            energy = preds
            loss = mae_loss(energy.reshape(-1), y)
        return loss, {}

    def to_encoder(self, output_dim=None):
        for output_block in self.model.output_blocks:
            if output_dim is None or output_dim == 256:
                output_block.lin = Identity()
                output_block.to_encoder()
            elif output_dim == 1:
                continue
            else:
                output_block.lin = Linear(256, output_dim)
        self.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                if self.is_encoder:
                    if name in re.findall('output_blocks\..\.lin\..*', name):
                        print('skipping freezing of parameter ', name)
                        continue
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False


class DimeNetPP_Alt(LightningInterface, BaseModel):
    def __init__(self, dimenet_pp_params: dict):
        super(DimeNetPP_Alt, self).__init__()
        self.model = DimeNetPP_Alternative(**dimenet_pp_params)
        self.is_encoder = False

    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)

    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        
        preds = self.model(batch.z, batch.pos, batch.batch)
        if self.model.predict_forces:
            energy, forces = preds
            loss = force_loss(energy.reshape(-1), forces, batch.energy, batch.force, rho_force=0.99)
        else:
            if batch.y is not None:
                y = batch.y
            elif batch.energy is not None:
                y = batch.energy
            else:
                raise NotImplementedError
            energy = preds
            loss = mae_loss(energy.reshape(-1), y)
        return loss, {}

    def to_encoder(self, output_dim=None):
        self.model.lin = Identity()
        self.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                if self.is_encoder:
                    if name in re.findall('output_blocks\..\.lin\..*', name):
                        print('skipping freezing of parameter ', name)
                        continue
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False

class NoModel(LightningInterface, BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.layer = Identity()

    def forward(self, batch):
        x, y = batch.x, batch.y
        return self.layer(x)
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        y = batch[1]
        output = self.layer(batch)
        mae = mae_loss(output, y)
        return mae, {}
    
    def to_encoder(self, **kwargs):
        print('just identity operation')
    
    def freeze(self, **kwargs):
        print('just identity operation')
        
class NormalizedModel(LightningInterface, BaseModel):
    def __init__(self, model: BaseModel, encoder_dim: int):
        super().__init__()
        self.model = model
        self.bn = torch.nn.BatchNorm1d(num_features=encoder_dim)

    def forward(self, x):
        return self.bn(self.model(x))
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        output = self.forward(batch)
        mae = mae_loss(output, batch[1])
        return mae, {}
    
    def freeze(self, param_list=None):
        return self.model.freeze(param_list)


class SchNet(LightningInterface, BaseModel):
    def __init__(self, params):
        super(SchNet, self).__init__()
        self.model = SchnetBase(**params)
        self.is_encoder = False
        
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)

    def to_encoder(self, output_dim=None):
        self.model.lin2 = Identity()
        self.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False

class SchNetDropout(LightningInterface, BaseModel):
    def __init__(self, params):
        super(SchNetDropout, self).__init__()
        assert "drop_prob" in params, "You did not specify a dropout probability for SchNet Dropout"
        self.model = SchNetDropoutBase(**params)
        self.is_encoder = False
        
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)

    def to_encoder(self, output_dim=None):
        self.model.lin2 = Identity()
        self.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False

class PaiNN(LightningInterface, BaseModel):
    def __init__(self, params):
        super(PaiNN, self).__init__()

        radial_basis = GaussianRBF(n_rbf=params['radial_basis'], cutoff=params['cutoff'])
        cutoff_fn = CosineCutoff(cutoff=params['cutoff'])
        pain_params = params.copy()
        pain_params.update({
            'radial_basis': radial_basis, 
            'cutoff_fn': cutoff_fn
        })
        self.model = PaiNN_Base(**pain_params)
        self.out_module = Atomwise(n_in=params['n_atom_basis'])
        self.is_encoder = False
        
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        data = self.model(z, pos, b)
        out_data = self.out_module(data)
        return out_data['y']
    
    def to_encoder(self):
        # self.out_module.outnet[0].activation = Identity()
        # self.out_module.outnet[1] = Identity()
        # self.is_encoder=True
        self.out_module.outnet = Identity()
    
    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False


class SphereNet(LightningInterface, BaseModel):
    def __init__(self, params):
        super(SphereNet, self).__init__()
        self.model = SphereNetBase(**params)
        self.is_encoder = False

    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)
