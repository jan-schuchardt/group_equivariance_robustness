from equivariance_robustness.attributes_zuegner.models import RobustGCNModelGED
import numpy as np
from numpy.typing import NDArray
from robust_gcn.robust_gcn import chunker, sparse_tensor
import scipy.sparse as sp

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def certify_ged_ibp(gcn_model: RobustGCNModelGED, attrs: sp.spmatrix,
                    q: int, nodes: NDArray[np.int64] = None, Q: int = 12, batch_size: int = 8,
                    certify_nonrobustness: bool = False,
                    progress: bool = False,
                    cost_add: float = 1, cost_del: float = 1, apply_relu: bool = False
                    ) -> tuple[NDArray[np.bool8], NDArray[np.bool8]]:
    """
    Certify (non-) robustness of the input nodes given the input GCN and attributes using IBP.

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
    apply_relu: bool
        Whether to apply ReLu to elementwise lower/upper bound after each IBP propagation step.

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

    if nodes is None:
        nodes = np.arange(N)

    lower_bounds = []
    nonrobust_nodes = np.zeros(N)
    if certify_nonrobustness:
        raise NotImplementedError('Certifying non-robustness not implemented yet.')

    _iter = chunker(np.arange(N), batch_size)
    if progress:
        _iter = tqdm(_iter, total=int(np.ceil(N/batch_size)))

    for chunk in _iter:
        lb = gcn_model.ibp(node_attrs, chunk, q, Q,
                           cost_add=cost_add, cost_del=cost_del,
                           apply_relu=apply_relu)

        lb = lb.detach()
        lower_bounds.append(lb.cpu().numpy())

    lower_bounds = np.row_stack(lower_bounds)
    robust_nodes = ((lower_bounds > 0).sum(1) >= K-1)

    return robust_nodes, nonrobust_nodes
