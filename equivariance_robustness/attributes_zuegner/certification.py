import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
import torch
from equivariance_robustness.attributes_zuegner.models import RobustGCNModelGED
from robust_gcn.robust_gcn import chunker, sparse_tensor
from torch import optim

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def certify_ged(gcn_model: RobustGCNModelGED, attrs: sp.spmatrix,
                q: int, nodes: NDArray[np.int64] = None, Q: int = 12,
                optimize_omega: bool = False, optimize_steps: int = 5, batch_size: int = 8,
                certify_nonrobustness: bool = False, progress: bool = False,
                cost_add: float = 1, cost_del: float = 1
                ) -> tuple[NDArray[np.bool8], NDArray[np.bool8]]:
    """
    Certify (non-) robustness of the input nodes given the input GCN and attributes.

    Generalizes robust_gcn.robust_gcn.certify to enable certification with variable costs for
    insertion and deletion.
    I.e., uses convex outer adversarial polytope, see Appendix F.3.
    Code is essentially the same, just calls RobustGCNModelGED-specific certification functions.

    Parameters
    ----------
    gcn_model: RobustGCNModel
        The input neural network.
    attrs: sp.spmatrix, [N, D]
        The binary node attributes.
    q: int
        The number of allowed perturbations per node.
    nodes: np.array, int64
        The input node indices to compute certificates for.
    Q: int
        The number of allowed perturbations globally.
    optimize_omega: bool, default False
        Whether to optimize (True) over Omega or to use the default value (False).
        If True, optimization takes significantly longer but will lead to more certificates.
        False positives (i.e. falsely issued certificates) are never possible.
    optimize_steps: int
        The number of steps to optimize Omega for. Ignored if optimize_omega is False.
    batch_size: int
        The batch size to use. Larger means faster computation but requires more GPU memory.
    certify_nonrobustness: bool, default: False
        Whether to also certify non-robustness. This works by determining the optimal perturbation
        for the relaxed GCN and feeding it into the original GCN. If this perturbation changes the
        predicted class, we have certified non-robustness via an example.
    progress: bool, default: False
        Whether to display a progress bar using the package `tqdm`. If it is not installed,
        we silently ignore this parameter.
    cost_add: float
            Cost of inserting a node attribute (0 --> 1).
            Uses up local budgets q and global budget Q.
    cost_del: float
        Cost of deleting a node attribute (1 --> 0).
        Uses up local budgets q and global budget Q.

    Returns
    -------
    robust_nodes: np.array, bool, [N,]
        A boolean flag for each of the input nodes indicating whether a robustness certificate
        can be issued.
    nonrobust_nodes: np.array, bool, [N,]
        A boolean flag for each of the input nodes indicating whether we can prove non-robustness.
        If certify_nonrobustness is False, this contains False for every entry.
    """

    node_attrs = sparse_tensor(attrs).cuda().to_dense()

    N = gcn_model.N
    K = gcn_model.K

    if optimize_omega:
        opt_omega = optim.Adam([{'params': x, "weight_decay": 0} for x in gcn_model.omegas])
    else:
        optimize_steps = 0

    if nodes is None:
        nodes = np.arange(N)

    for step in range(optimize_steps):
        for chunk in chunker(nodes, batch_size):

            obj = gcn_model.dual_backward(node_attrs, chunk, q, Q,
                                          initialize_omega=(step==0),
                                          optimize_omega=True)

            margin_loss = (-obj.min(1)[0].mean())
            with torch.no_grad():
                margin_loss.backward()
            opt_omega.step()
            opt_omega.zero_grad()

    lower_bounds = []

    nonrobust_nodes = np.zeros(N)

    _iter = chunker(np.arange(N), batch_size)
    if progress:
        _iter = tqdm(_iter, total=int(np.ceil(N/batch_size)))

    for chunk in _iter:
        # Only difference to original code is calling
        # dual_backward_ged instead of dual_backward here
        lb, pert = gcn_model.dual_backward_ged(node_attrs, chunk, q, Q,
                                           initialize_omega=not optimize_omega,
                                           return_perturbations=False, #True,
                                           cost_add=cost_add, cost_del=cost_del)
        lb = lb.detach()
        lower_bounds.append(lb.cpu().numpy())
        if certify_nonrobustness:
            raise NotImplementedError('Adversarial attacks not supported yet.')

    lower_bounds = np.row_stack(lower_bounds)
    robust_nodes = ((lower_bounds > 0).sum(1) == K-1)

    return robust_nodes, nonrobust_nodes