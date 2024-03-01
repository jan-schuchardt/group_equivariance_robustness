import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
import torch
from robust_gcn.robust_gcn import (RobustGCNLayer, RobustGCNModel,
                                   preprocess_adj, sparse_tensor)
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import relu

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


class RobustGCNLayerGED(RobustGCNLayer):
    """
    GCN layer that works as a normal layer in the forward pass
    but also provides a backward pass through the dual network.

    Generalizes RobustGCNLayer to enable certification with variable costs for
    insertion and deletion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bounds_binary_ged(self, input: torch.BoolTensor, nodes: NDArray[np.int64],
                      q: int, Q: int, lower_bound: bool = True, slice_input: bool = False,
                      cost_add: float = 1, cost_del: float = 1) -> torch.FloatTensor:
        """
        Compute bounds on the first layer for binary node attributes.

        Generalizes RobustGCNLayer.bounds_binary to enable variable costs for insertion and deltion.
        This corresponds to the first-layer bound from our Appendix F.2.

        Parameters
        ----------
        input: torch.tensor (boolean) dimension [Num. L-1 hop neighbors, D]
            binary node attributes (one vector for all neighbors of the input nodes)
            OR: [N, D] for the whole graph when slice_input=True
        nodes:  numpy.array, int64 dim [Num l hop neighbors,]
            L-l hop neighbors of the target nodes.
        q:  int
            per-node constraint on the number of attribute perturbations
        Q:  int
            global constraint on the number of attribute perturbations
        lower_bound: bool
            Whether to compute the lower bounds (True) or upper bounds (False)
        slice_input: bool
            Whether the input is the whole attribute matrix. If True, we slice the
            node features accordingly.
        cost_add: float
            Cost of inserting a node attribute (0 --> 1).
            Uses up local budgets q and global budget Q.
        cost_del: float
            Cost of deleting a node attribute (1 --> 0).
            Uses up local budgets q and global budget Q.

        Returns
        -------
        bounds: torch.tensor (float32) dimension [Num. L-2 hop neighbors x H_2]
            Lower/upper bounds on the hidden activations in the second layer.
        """

        # Convention:
        # N: number of nodes in the current layer
        # N_nbs: number of neighbors of the nodes in the current layer
        # D: dimension of the node attributes (i.e. H_1)
        # H: dimension of the first hidden layer (i.e. H_2)

        adj_slice, nbs = self.slice_adj(nodes)
        N_nbs = len(nbs)
        N = len(nodes)
        if slice_input:
            input = input[nbs]

        # [N_nbs x D] => [N_nbs x D x 1]
        input_extended = input.unsqueeze(2)

        # get the positive and negative parts of the weights
        # [D x H]  => [1 x D X H]
        W_plus = F.relu(self.weights).unsqueeze(0)
        W_minus = F.relu(-self.weights).unsqueeze(0)

        # Value matrix V in Algorithm 3
        # [N_nbs x D x H]
        if lower_bound:
            bounds_nbs = input_extended.mul(W_plus) + (1 - input_extended).mul(W_minus)
        else:
            bounds_nbs = (1 - input_extended).mul(W_plus) + input_extended.mul(W_minus)

        # Go from absolute values to values relative to cost
        assert torch.all((input_extended == 1) | (input_extended == 0))

        costs = torch.zeros_like(bounds_nbs)  # [N_nbs x D x H]
        costs[input == 1, :] = cost_del
        costs[input == 0, :] = cost_add

        # [N_nbs x D x H]
        # Ratio matrix R in Algorithm 3
        bounds_nbs_ratio = torch.clone(bounds_nbs) / costs

        # top p entries per dimension in D, with p chosen such that budget q definitely used up.
        # Essentially corresponds to argsort_desc(R_n) in Algorithm 3
        # => [N_nbs x p x H]
        max_changes_local = int(np.ceil(q / min(cost_add, cost_del)))  # This is p in comments

        top_p_ratios, top_p_idx = bounds_nbs_ratio.topk(max_changes_local, 1)
        top_p_costs = torch.gather(costs, 1, top_p_idx)  # sort costs like ratios

        """ Allocate perturbations until reaching local budget q
        Corresponds to first loop in Algorithm 3
        => [N_nbs x p x H]
        """

        # How much budget is spent after perturbing the 0, 1, 2, ..., bits with the highest ratio
        # can use cumsum, because topk already sorts in descending order
        max_budget_spent_local = torch.cat([torch.zeros((top_p_costs.shape[0], top_p_costs.shape[2])).unsqueeze(1).cuda(),
                                            torch.cumsum(top_p_costs, 1)],
                                            dim=1)[:, :-1, :]

        # How much budget is left after making this allocation
        local_budget_left = torch.clamp(q - max_budget_spent_local, min=0)
        # L from paper, i.e., whether bit is (partially) perturbed
        allocation_local = torch.clamp(local_budget_left / top_p_costs, max=1)

        # => [N_nbs x p*H]
        top_p_ratios = top_p_ratios.reshape([N_nbs, -1])

        # [N x N_nbs x 1]
        adj_extended = adj_slice.unsqueeze(2).to_dense()

        # per-node bounds (after aggregating the neighbors)
        # [N x N_nbs x p x H]
        aggregated_ratios = adj_extended.mul(top_p_ratios).reshape([N, N_nbs, max_changes_local, -1])
        allocation_local = allocation_local[None, ...].repeat_interleave(N, 0)
        top_p_costs = top_p_costs[None, ...].repeat_interleave(N, 0)

        """ For each Node n in \{1,...,N\},
        sum up the top P values of the top p values per dimension.
        Corresponds to second loop in Algorith 3
        [N, P, H] => [N, H]
        """

        max_changes_global = int(np.ceil(Q / min(cost_add, cost_del)))
        # Number of bits that can be perturbed is bounded by global budget and local budgets of all neighbors
        n_sel = min(N_nbs * max_changes_local, max_changes_global)  # This is P in comments

        aggregated_ratios = aggregated_ratios.reshape([N, -1, self.H])
        allocation_local = allocation_local.reshape([N, -1, self.H])
        top_p_costs = top_p_costs.reshape([N, -1, self.H])

        # This corresponds to argsort_desc(R) in Algorithm 3
        top_P_ratios, top_P_idx = aggregated_ratios.topk(n_sel, 1)
        top_P_allocation_local = torch.gather(allocation_local, 1, top_P_idx)  # sort
        top_P_costs = torch.gather(top_p_costs, 1, top_P_idx)  # sort

        # C* x Q* from Algorithm 3
        effective_costs = top_P_allocation_local * top_P_costs

        max_budget_spent_global = torch.cat([torch.zeros((effective_costs.shape[0], effective_costs.shape[2])).unsqueeze(1).cuda(),
                                             torch.cumsum(effective_costs, 1)],
                                             dim=1)[:, :-1, :]

        global_budget_left = torch.clamp(Q - max_budget_spent_global, min=0)
        # min(L, Q*) from Algorithm 3
        allocation_global = torch.minimum(global_budget_left / top_P_costs, top_P_allocation_local)

        # V x Q from Algorithm 3
        change = (allocation_global * top_P_ratios * top_P_costs).sum(1)

        if lower_bound:
            change *= -1

        # Add the normal hidden activations for the input
        bounds = change + self.forward(input, nodes)
        return bounds


class RobustGCNModelGED(RobustGCNModel):
    """
    GCN model that works as a normal one in the forward pass
    but also enables robustness certification and robust training
    via the backward pass through the dual network.

    Generalizes RobustGCNLayer to enable variable costs for insertion and delketion.
    """

    def __init__(self, adj, dims):
        super().__init__(adj, dims)
        adj_prep = preprocess_adj(adj).tocsr()
        self.adj_norm = adj_prep
        self.layers = []
        self.dims = dims
        self.K = int(dims[-1])
        self.N = self.adj_norm.shape[0]

        self.omegas = []
        previous = dims[0]  # data dimension
        for ix,hidden in enumerate(dims[1:]):
            self.layers.append(RobustGCNLayerGED(self.adj_norm, [previous, hidden]))
            self.add_module(f"conv:{ix}", self.layers[-1])
            previous = hidden
            if ix + 2 < len(dims):
                self.omegas.append(torch.zeros([self.N, dims[ix+1]], requires_grad=True))

    def dual_backward_ged(self, input: torch.Tensor, nodes: NDArray[np.int64],
                          q: int, Q: int, target_classes: torch.LongTensor | None = None,
                          initialize_omega: bool = False, optimize_omega: bool = False,
                          return_perturbations: bool = False,
                          cost_add: float = 1, cost_del: float = 1) -> torch.FloatTensor:
        """
        Backward computation through the "dual network" to get lower bounds
        on the worst-case logit margins achievable given the provided local
        and global constraints on the perturbations.

        Generalizes RobustGCNModelGED.dual_backward to enable variable costs.
        As shown in Appendix F.3, this is basically identical to the original procedure,
        safe for the computation of lambda and o (here called rho, eta),
        and a cost factor in the computation of Psi.

        Parameters
        ----------
        input: torch.tensor float32 or int, dim [N, D]
            The binary node attributes.
        nodes: numpy.array, int64
            The input nodes for which to compute the worst-case margins.
        q:  int
            per-node constraint on the number of attribute perturbations
        Q:  int
            global constraint on the number of attribute perturbations
        target_classes: torch.tensor, int64, dim [B,] or None
            The target classes of the nodes in the batch. For nodes in the training set,
            this should be the correct (known) class. For the unlabeled nodes, this should be
            the predicted class given the current weights.
        initialize_omega: bool
            Whether the omega matrices should be initialized to their default value,
            which is upper_bound/(upper_bound-lower_bound). This is only relevant for
            robustness certification (not for robust training, which always uses
            the default values for omega).
        cost_add: float
            Cost of inserting a node attribute (0 --> 1).
            Uses up local budgets q and global budget Q.
        cost_del: float
            Cost of deleting a node attribute (1 --> 0).
            Uses up local budgets q and global budget Q.

        Returns
        -------
        worst_case_bounds: torch.tensor float32, dim [len(nodes), K]
            Lower bounds on the worst-case logit margins achievable given the input constraints.
            A negative worst-case logit margin lower bound means that we cannot certify robustness.
            A positive worst-case logit margin lower bound guarantees that the prediction will not
            change, i.e. we can issue a robustness certificate if for a node ALL worst-case
            logit margins are positive.
        """
        if not (torch.sort(input.unique().long().cpu())[0] == torch.tensor([0,1])).all():
            raise ValueError("Node attributes must be binary.")

        input = input.float()

        # compute upper/lower bounds first
        batch_size = len(nodes)
        bounds = []
        neighborhoods = self.get_neighborhoods(nodes)[::-1]
        for ix, layer in enumerate(self.layers[:-1]):
            layer = self.layers[ix]
            nbh = neighborhoods[ix]
            if ix == 0:
                lower_bound = layer.bounds_binary_ged(input, nbh, q, Q, slice_input=True,
                                                  lower_bound=True, cost_add=cost_add, cost_del=cost_del)
                upper_bound = layer.bounds_binary_ged(input, nbh, q, Q, slice_input=True,
                                                  lower_bound=False, cost_add=cost_add, cost_del=cost_del)
                bounds.append((lower_bound, upper_bound))
            else:
                bounds.append(tuple(layer.bounds_continuous(bounds[-1][0], bounds[-1][1],
                                                      nbh, slice_input=False)))

        if target_classes is None:
            # if no target classes are supplied, we use the current predictions
            target_classes = self.predict(input, nodes, )

        predicted_onehot = torch.eye(self.K)[target_classes]
        # [Batch, K, K]
        C_tensor = (predicted_onehot.unsqueeze(1) - torch.eye(self.K)).cuda()
        phis = [-C_tensor]

        # final_objective = torch.zeros([batch_size, self.K], device="cuda")
        bias_terms = torch.zeros([batch_size, self.K], device="cuda")
        I_terms = torch.zeros([batch_size, self.K], device="cuda")

        for ix in np.arange(1,len(self.layers))[::-1]:
            layer = self.layers[ix]
            phi = phis[-1]
            nodes = neighborhoods[ix]

            compute_objective = ix > 0
            is_last_layer = ix == len(self.layers) - 1

            if optimize_omega:
                if initialize_omega:
                    lb, ub = bounds[ix - 1]
                    I = ((ub>0) & (lb < 0)).float()
                    omega = (ub / (ub-lb + 1e-9)).mul(I).clone().detach().requires_grad_(True)
                    with torch.no_grad():
                        self.omegas[ix-1].index_put_((torch.LongTensor(neighborhoods[ix-1]),),
                                                     omega.cpu())
                omega = self.omegas[ix - 1][neighborhoods[ix - 1]].cuda()
            else:
                omega = None
            ret = layer.dual_backward(phi, nodes, bounds[ix - 1], is_last_layer,
                                      compute_objective, omega=omega)

            next_phi, bias_term, objective_term = ret
            phis.append(next_phi)
            bias_terms += bias_term
            if objective_term is not None:
                I_terms += objective_term

        # get the L-2 hop neighbors of the target nodes
        # e.g. the 1-hop neighbors for a 3-layer GCN (i.e. one hidden layer)
        nodes_first = neighborhoods[0]
        phi_1_hat = self.layers[0].phi_backward(phis[-1], nodes=nodes_first)
        nbs_first = self.get_neighbors(nodes_first)

        # sum up the bias terms of the layers
        bias_terms += self.layers[0].bias_objective_term(phis[-1])  # [B, K]

        # [B, K, Num. L-1 hop neighbors, D]
        Delta = relu(phi_1_hat).mul(1 - input[nbs_first]) \
              + relu(-phi_1_hat).mul(input[nbs_first])
        costs = torch.zeros_like(Delta)  # [B, K, Num. L-1 hop neighbors, D]
        costs[:, :, input[nbs_first] == 1] = cost_del
        costs[:, :, input[nbs_first] == 0] = cost_add

        """ Find worst-case global and local budget allocation,
        so we can compute lambda and o from Appendix F.3.
        Procedure is basically identical to that in bounds_binary_ged above."""

        # First loop from Algorithm 3
        max_changes_local = int(np.ceil(q / min(cost_add, cost_del)))  # p from comments
        top_p_ratios, top_p_idx = (Delta / costs).topk(max_changes_local, 3)
        top_p_costs = torch.gather(costs, 3, top_p_idx)  # sort costs like ratios

        max_budget_spent_local = torch.cat([torch.zeros(top_p_costs.shape[:3]).unsqueeze(3).cuda(),
                                            torch.cumsum(top_p_costs, 3)],
                                            dim=3)[..., :-1]

        local_budget_left = torch.clamp(q - max_budget_spent_local, min=0)
        allocation_local = torch.clamp(local_budget_left / top_p_costs, max=1)

        p_largest_local = top_p_ratios.clone()
        p_largest_local[allocation_local == 0] = torch.inf
        p_largest_local = p_largest_local.min(dim=3).values

        # [B, K, (Num L-1 hop neighbors) * p]
        top_p_ratios_overall = top_p_ratios.reshape([batch_size, self.K, -1])
        top_p_costs_overall = top_p_costs.reshape([batch_size, self.K, -1])
        allocation_local_overall = allocation_local.reshape([batch_size, self.K, -1])

        # Second loop from Algorithm 3
        max_changes_global = int(np.ceil(Q / min(cost_add, cost_del)))  # [B, K, P]
        n_sel = min(len(nbs_first) * max_changes_local, max_changes_global)  # P from comments

        top_P_ratios, top_P_idx = top_p_ratios_overall.topk(n_sel, 2)
        top_P_allocation_local = torch.gather(allocation_local_overall, 2, top_P_idx)  # sort
        top_P_costs = torch.gather(top_p_costs_overall, 2, top_P_idx)  # sort

        effective_costs = top_P_allocation_local * top_P_costs

        max_budget_spent_global = torch.cat([torch.zeros(effective_costs.shape[:2]).unsqueeze(2).cuda(),
                                             torch.cumsum(effective_costs, 2)],
                                             dim=2)[..., :-1]

        global_budget_left = torch.clamp(Q - max_budget_spent_global, min=0)
        allocation_global = torch.minimum(global_budget_left / top_P_costs, top_P_allocation_local)

        top_P_ratios[allocation_global == 0] = torch.inf
        # Select the P-th largest element of the p-th largest elements
        # This corresponds to lambda from our appendix F.3
        rho = top_P_ratios.min(dim=2).values.unsqueeze(2)  # [B, K, 1]

        # Indices of the perturbations
        if return_perturbations:
            raise NotImplementedError('Adversarial attacks not implemented yet.')

        # Select the smallest of the p largest values per node, or 0 if it is smaller than rho.
        # This corresponds to o_n from our Appendix F.3
        eta = relu(p_largest_local - rho)  # [B, K, Num L-1 hop neighbors]

        # Compute Psi (c.f. the paper) and sum over it
        # Note that we multiply with costs, see Appendix F.3.
        Psi_term = relu(Delta - costs * (rho + eta).unsqueeze(-1)).abs().sum((2, 3))  # [B, K]
        trace_term = input[nbs_first].mul(phi_1_hat).sum((2, 3))  # [B, K]

        # [B, K] lower-bound worst-case margins w.r.t. all other classes
        final_objective = I_terms - bias_terms  - trace_term  - Psi_term - q * eta.sum(-1) - Q * rho.squeeze(-1)

        return final_objective, None


    def ibp(self, input: torch.Tensor, nodes: NDArray[np.int64], q: int, Q: int,
            target_classes: torch.LongTensor | None = None,
            cost_add: float = 1, cost_del: float = 1, apply_relu: bool = False) -> torch.FloatTensor:
        """Interval bound propagation from Appendix F.2.

        First solves knapsack problem for first layer via Algorithm 3,
        then uses standard IBP, as implemented in RobustGCNLayerGED.bounds_continuous.

        Args:
            input: torch.tensor float32 or int, dim [N, D]
                The binary node attributes.
            nodes: numpy.array, int64
                The input nodes for which to compute the worst-case margins.
            q:  int
                per-node constraint on the number of attribute perturbations
            Q:  int
                global constraint on the number of attribute perturbations
            target_classes: torch.tensor, int64, dim [B,] or None
                The target classes of the nodes in the batch. For nodes in the training set,
                this should be the correct (known) class. For the unlabeled nodes, this should be
                the predicted class given the current weights.
            cost_add: float
                Cost of inserting a node attribute (0 --> 1).
                Uses up local budgets q and global budget Q.
            cost_del: float
                Cost of deleting a node attribute (1 --> 0).
                Uses up local budgets q and global budget Q.
            apply_relu: bool
                Whether to apply ReLU to upper and lower bound after each layer.

        Returns:
            worst_case_bounds: torch.tensor float32, dim [len(nodes), K]
                Lower bounds on the worst-case logit margins achievable given the input constraints.
                A negative worst-case logit margin lower bound means that we cannot certify robustness.
                A positive worst-case logit margin lower bound guarantees that the prediction will not
                change, i.e. we can issue a robustness certificate if for a node ALL worst-case
                logit margins are positive.
        """

        if not (torch.sort(input.unique().long().cpu())[0] == torch.tensor([0,1])).all():
            raise ValueError("Node attributes must be binary.")

        input = input.float()

        # compute upper/lower bounds first
        batch_size = len(nodes)
        bounds = []
        neighborhoods = self.get_neighborhoods(nodes)[::-1]
        for ix, layer in enumerate(self.layers):
            layer = self.layers[ix]
            nbh = neighborhoods[ix]
            apply_relu = apply_relu and (ix != len(self.layers) - 1)
            if ix == 0:
                lower_bound = layer.bounds_binary_ged(input, nbh, q, Q, slice_input=True,
                                                  lower_bound=True, cost_add=cost_add, cost_del=cost_del)
                upper_bound = layer.bounds_binary_ged(input, nbh, q, Q, slice_input=True,
                                                  lower_bound=False, cost_add=cost_add, cost_del=cost_del)
                bounds.append((lower_bound, upper_bound))
            else:
                lower_bound, upper_bound = layer.bounds_continuous(bounds[-1][0], bounds[-1][1],
                                                      nbh, slice_input=False)

            if apply_relu:
                lower_bound = F.relu(lower_bound)
                upper_bound = F.relu(upper_bound)

            bounds.append((lower_bound, upper_bound))

        if target_classes is None:
            # if no target classes are supplied, we use the current predictions
            target_classes = self.predict(input, nodes, )

        # [Batch, K, K]
        logits_lower, logits_upper = bounds[-1]
        margins = logits_lower[:, :, None] - logits_upper[:, None, :]

        # [B, K] lower-bound worst-case margins w.r.t. all other classes
        margins = margins[torch.arange(batch_size), target_classes, :]

        return margins
