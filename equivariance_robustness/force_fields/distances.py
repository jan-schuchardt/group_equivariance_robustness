import torch


def average_l2(x, Y):
    """
    x: [N x D]
    Y: [B x N x D]

    return: [B, ]
    """

    return torch.mean(torch.norm((Y - x), p=2, dim=2), dim=1)
