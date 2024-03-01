"""
    This module contains the training code of the graph_cert reference implementation [1],
    modified to our framework.
    [1] https://github.com/abojchevski/graph_cert
"""
import numpy as np
import torch
import torch.nn.functional as F


def train(model, attr, ppr, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50):
    """Train a model using either standard or adversarial training.

    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    ppr: torch.Tensor [n, n]
        Dense Personalized PageRank matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    adver_config : dict
        Dictionary encoding the parameters for adversarial training.

    Returns
    -------
    trace_val: list
        A list of values of the validation loss during training.
    """

    trace_val = []
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf
    best_acc = -np.inf

    for it in range(max_epochs):
        logits, diffused_logits = model(attr=attr, ppr=ppr)

        # standard cross-entropy
        p_robust = -1
        loss_train = ce_loss(
            diffused_logits=diffused_logits[idx_train], labels=labels[idx_train])
        loss_val = ce_loss(
            diffused_logits=diffused_logits[idx_val], labels=labels[idx_val])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        trace_val.append(loss_val.item())

        acc_train = accuracy(labels, diffused_logits, idx_train)
        acc_val = accuracy(labels, diffused_logits, idx_val)

        if loss_val < best_loss or acc_val > best_acc:
            best_loss = loss_val
            best_acc = acc_val
            best_epoch = it
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            print(f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                  f' acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} p_robust {p_robust:.5f}'
                  )

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_val


def ce_loss(diffused_logits, labels):
    """Compute the standard cross-entropy loss.

    Parameters
    ----------
    diffused_logits: torch.Tensor, [?, nc]
        Logits diffused by Personalized PageRank.
    labels: torch.Tensor [?]
        The ground-truth labels.

    Returns
    -------
    loss: torch.Tensor
        Standard cross-entropy loss.
    """
    return F.cross_entropy(diffused_logits, labels)


def rce_loss(adv_logits, labels):
    """Compute the robust cross-entropy loss.

    Parameters
    ----------
    adv_logits: torch.Tensor, [?, nc]
        The worst-case logits for each class for a batch of nodes.
    labels: torch.Tensor [?]
        The ground-truth labels.

    Returns
    -------
    loss: torch.Tensor
        Robust cross-entropy loss.
    """
    return F.cross_entropy(-adv_logits, labels)


def cem_loss(diffused_logits, adv_logits, labels, margin):
    """ Compute the robust hinge loss.

    Parameters
    ----------
    diffused_logits: torch.Tensor, [?, nc]
        Logits diffused by Personalized PageRank.
    adv_logits: torch.Tensor, [?, nc]
        The worst-case logits for each class for a batch of nodes.
    labels: torch.Tensor [?]
        The ground-truth labels.
    margin : int
        Margin.

    Returns
    -------
    loss: torch.Tensor
        Robust hinge loss.
    """
    hinge_loss_per_instance = torch.max(
        margin - adv_logits, torch.zeros_like(adv_logits)).sum(1) - margin
    loss_train = F.cross_entropy(
        diffused_logits, labels, reduction='none') + hinge_loss_per_instance
    return loss_train.mean()


def accuracy(labels, logits, idx):
    """ Compute the accuracy for a set of nodes.

    Parameters
    ----------
    labels: torch.Tensor [n]
        The ground-truth labels for all nodes.
    logits: torch.Tensor, [n, nc]
        Logits for all nodes.
    idx: array-like [?]
        The indices of the nodes for which to compute the accuracy .

    Returns
    -------
    accuracy: float
        The accuracy.
    """
    return (labels[idx] == logits[idx].argmax(1)).sum().item() / idx.sum()
