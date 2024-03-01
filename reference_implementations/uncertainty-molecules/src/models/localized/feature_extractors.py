import torch
from torch.nn import Identity
from typing import Tuple
from torch_geometric.nn import DimeNet as DimeNet_Geom
from src.models.standard.schnet import SchNet as SchnetBase
from src.models.standard.schnet_dropout import SchNetDropout as SchNetDropoutBase
from src.models.interfaces import BaseModel, LightningInterface
from src.models.standard.dimenet_pp_dropout import  DimeNetPPDropout as DimenetDropoutBase
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from src.models.standard.dimenet_pp import DimeNetPP
from src.utils import torch_mae as mae_loss
import re
from torch.nn import Linear


class SchNetMult(LightningInterface, BaseModel):
    def __init__(self, params):
        super(SchNetMult, self).__init__()
        self.model = SchnetBase(**params, localized=True)
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

class SchNetMultDropout(LightningInterface, BaseModel):
    def __init__(self, params):
        super(SchNetMultDropout, self).__init__()
        self.model = SchNetDropoutBase(**params, localized=True)
        self.is_encoder = False
        
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)
 
    def to_encoder(self, output_dim=None):
        self.model.lin2 = Identity()
        self.is_encoder = True
        self.model.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False

class DimeNetMulti(LightningInterface, BaseModel):
    def __init__(self, dimenet_params):
        super(DimeNetMulti, self).__init__()
        self.model = DimeNetMultiBase(dimenet_params)
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


class DimeNetPPMultiDropout(LightningInterface, BaseModel):
    def __init__(self, dimenet_pp_params):
        super(DimeNetPPMultiDropout, self).__init__()
        self.model = DimeNetPPDropoutMultiBase(dimenet_pp_params)
        self.is_encoder = False
    
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        preds = self.forward(batch.z, batch.pos, batch.batch)
        mae = mae_loss(preds, batch.y)
        return mae, {}

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
                
class DimeNetPPMulti(LightningInterface, BaseModel):
    def __init__(self, dimenet_params):
        super(DimeNetPPMulti, self).__init__()
        self.model = DimeNetPPMultiBase(dimenet_params)
        self.is_encoder = False
    
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)
    
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        preds = self.forward(batch.z, batch.pos, batch.batch)
        mae = mae_loss(preds, batch.y)
        return mae, {}

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
        
class DimeNetMultiBase(DimeNet_Geom):
    def __init__(self, dimenet_params: dict):
        super(DimeNetMultiBase, self).__init__(**dimenet_params)
    
    def forward(self, z, pos, batch=None):
            """"""
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                    max_num_neighbors=self.max_num_neighbors)

            i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
                edge_index, num_nodes=z.size(0))

            # Calculate distances.
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

            # Calculate angles.
            pos_i = pos[idx_i]
            pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
            angle = torch.atan2(b, a)

            rbf = self.rbf(dist)
            sbf = self.sbf(dist, angle, idx_kj)

            # Embedding block.
            x = self.emb(z, rbf, i, j)
            P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

            # Interaction blocks.
            for interaction_block, output_block in zip(self.interaction_blocks,
                                                    self.output_blocks[1:]):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
                P += output_block(x, rbf, i)

            return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
    
class DimeNetPPMultiBase(DimeNetPP):
    def __init__(self, dimenet_params: dict, natoms=None):
        super(DimeNetPPMultiBase, self).__init__(**dimenet_params)
        self.natoms = natoms
    
    def forward(self, z, pos, batch=None):
        """"""
        if batch is not None:
            batch_size = max(batch) + 1
        else: 
            batch_size = 1
        natoms = int(len(pos) / batch_size)

        if self.predict_forces:
            torch.set_grad_enabled(True)
            pos.requires_grad = True
            
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_kj = pos[idx_j] - pos_i, pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))


        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        #P = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        return P#.reshape(batch_size, natoms, -1)
    
    
class DimeNetPPDropoutMultiBase(DimenetDropoutBase):
    def __init__(self, dimenet_pp_params: dict):
        super(DimeNetPPDropoutMultiBase, self).__init__(**dimenet_pp_params)

    def forward(self, z, pos, batch=None):
        """"""
        if batch is not None:
            batch_size = max(batch) + 1
        else: 
            batch_size = 1
            
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_kj = pos[idx_j] - pos_i, pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        #P = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        return P
