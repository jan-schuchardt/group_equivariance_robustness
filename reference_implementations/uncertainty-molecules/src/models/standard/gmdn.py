from typing import Tuple, Optional, List

import torch
#from pydgn.model.interface import ReadoutInterface
from torch.distributions import Categorical, Independent, Binomial, Normal, MixtureSameFamily
from torch.nn import Identity, Linear, ReLU, Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Batch


class GMDN(torch.nn.Module):

    def __init__(self, dim_node_features, no_experts, encoder):
        super().__init__()
        self.device = None

        self.dim_node_features = dim_node_features
        self.no_experts = no_experts

        self.emission = GraphExpertEmission(dim_node_features, no_experts, 1, 10)

        self.hidden_units = 64
        self.dirichlet_alpha = 1.0
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.dirichlet_alpha] * self.no_experts,
                                                                              dtype=torch.float32))
        self.transition = GMDNTransition(dim_node_features, no_experts)  # neural convolution
        self.encoder = encoder

    def to(self, device):
        if device is not None:
            self.device = device
            self.emission.to(device)
            self.transition.to(device)
            self.dirichlet.concentration = self.dirichlet.concentration.to(device)

    def forward(self, data):

        embeddings = self.encoder(data)
        # Perform the neural aggregation of neighbors to return the posterior P(Q=i | graph)
        # weight sharing: pass node embeddings produced by the gating network (which is a DGN) to each expert
        mixing_weights, node_embeddings = self.transition(embeddings, data.batch)

        distr_params = self.emission.get_distribution_parameters(embeddings, data.batch)

        return distr_params, mixing_weights
    
    def get_std(self, data):
        dist_params, mixing_weights = self.forward(data)
        return (mixing_weights @ dist_params[1]).item()
    
    def get_sum_std(self, data):
        dist_params, mixing_weights = self.forward(data)
        return dist_params[1].sum()
    
    
class GraphExpertEmission(torch.nn.Module):
    """
    STRUCTURE AGNOSTIC EMISSION, uses node embeddings. Implements all experts
    """

    def forward(self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        pass

    def __init__(self, dim_features, no_experts, dim_target, expert_hidden_units, linear_final=False):
        super().__init__()

        self.no_experts = no_experts
        self.hidden_units = expert_hidden_units
        self.dim_target = dim_target
        self.linear_final = linear_final
        self.aggregate = global_add_pool

        self.output_activation = Identity()  # emulate gaussian (needs variance as well
        # Need independent parameters for the variance
        if self.hidden_units > 0:
            self.node_transform = Sequential(Linear(dim_features, self.hidden_units * self.no_experts * 2), ReLU())
            if self.linear_final:
                self.final_transform = Linear(self.hidden_units * self.no_experts * 2, self.no_experts * 2 * dim_target)
            else:
                self.final_transform = Sequential(
                    Linear(self.hidden_units * self.no_experts * 2, self.hidden_units * self.no_experts * 2), 
                    ReLU(),
                    Linear(self.hidden_units * self.no_experts * 2, self.hidden_units * self.no_experts * 2), 
                    ReLU(),
                    Linear(self.hidden_units * self.no_experts * 2, self.no_experts * 2 * dim_target)
                )
            
        else:
            self.node_transform = Identity()
            self.final_transform = Linear(dim_features, self.no_experts * 2 * dim_target)

    def get_distribution_parameters(self, node_embeddings, batch):

        if self.aggregate is not None:
            graph_embeddings = self.aggregate(self.node_transform(node_embeddings), batch)
            out = self.output_activation(self.final_transform(graph_embeddings))
        else:
            out = self.output_activation(self.final_transform(self.node_transform(node_embeddings)))

        # Assume isotropic gaussians
        params = torch.reshape(out, [-1, self.no_experts, 2, self.dim_target])  # ? x no_experts x 2 x F
        mu, var = params[:, :, 0, :], params[:, :, 1, :]

        var = torch.nn.functional.softplus(var) + 1e-8
        # F is assumed to be 1 for now, add dimension to F

        distr_params = (mu, var)   # each has shape ? x no_experts X F

        return distr_params




class GMDNTransition(torch.nn.Module):
    """
    Computes the vector of mixing weights
    """

    def __init__(self, hidden_units, dim_target):
        super().__init__()

        self.dim_target = dim_target

        self.aggregate = global_add_pool
        self.out = Linear(hidden_units, dim_target)

    def forward(self, x, batch):

        if self.aggregate is not None:
            x = self.aggregate(x, batch)
        out = self.out(x)

        # Exp-normalize trick: subtract the maximum value
        max_vals, _ = torch.max(out, dim=-1, keepdim=True)
        out_minus_max = out - max_vals
        mixing_weights = torch.nn.functional.softmax(out_minus_max, dim=-1).clamp(1e-8, 1.)
        return mixing_weights, x