import torch
from torch import nn
import numpy as np
from numpy.typing import NDArray
from torch import optim
from robust_gcn.robust_gcn import RobustGCNModel

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def train(gcn_model: RobustGCNModel, X: torch.Tensor, y: torch.LongTensor,
          idx_train: NDArray[np.int64], idx_val: NDArray[np.int64],
          n_iters: int = 3000, method="Normal",
          early_stopping: int = 50,
          learning_rate: float = 1e-3, weight_decay: float = 5e-4) -> None:
    """Trains model for specified number of epochs using Adam with early stopping.

    Args:
        gcn_model (RobustGCNModel): Model to train.
            Remember to specify adjacency matrix by setting "A" attribute
            before calling train().
        X (torch.Tensor): [N x D] attribute matrix in sparse COO format.
        y (torch.LongTensor): [N] target vector.
        idx_train (NDArray[np.int64]): Indices of train set.
            Values should be in {0,...,N-1}
        idx_val (NDArray[np.int64]): Indices of validation set.
            Values should be in {0,...,N-1}
        n_iters (int, optional): Number of gradient steps. Defaults to 3000.
        method (str, optional): Defaults to "Normal".
        early_stopping (int, optional): Number of gradient steps without improvement
            before stopping training prematurely. Defaults to 50.
        learning_rate (float, optional): The Adam learning rate. Defaults to 1e-3.
        weight_decay (float, optional): The Adam weight decay. Defaults to 5e-4.

    Raises:
        NotImplementedError: _description_
    """

    implemented_methods = ["Normal"]
    if method not in implemented_methods:
        raise NotImplementedError(f"Method not in {implemented_methods}.")

    if 'early_stopping' is not None:
        early_stopping = early_stopping
    else:
        early_stopping = np.inf

    params = list(gcn_model.parameters())
    weights = [p for p in params if p.requires_grad and len(p.shape) == 2]
    biases = [p for p in params if p.requires_grad and len(p.shape) == 1]

    param_list = [{'params': weights},
                  {'params': biases}]
    opt = optim.Adam(param_list, lr=learning_rate, weight_decay=weight_decay)

    best_loss = np.inf
    best_epoch = 0
    best_state = {}    

    tq = tqdm(range(n_iters))
    for it in tq:
        gcn_model.train()
        opt.zero_grad()

        logits = gcn_model(X)
        loss_train = nn.functional.cross_entropy(logits[idx_train], target=y[idx_train])
        with torch.no_grad():
            loss_train.backward()

        opt.step()

        gcn_model.eval()
        with torch.no_grad():
            logits = gcn_model(X)
            loss_val = nn.functional.cross_entropy(logits[idx_val], target=y[idx_val]).detach()

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu()
                          for key, value in gcn_model.state_dict().items()}

        if it - best_epoch > early_stopping:
            print(f"early stopping at epoch {it}")
            break

    gcn_model.load_state_dict(best_state)
