import torch
import torch.distributions as D
from math import sqrt
import numpy as np
import scipy


def _calibration_regression(pred, true, confidence):
    """
    :param pred: torch.distributions (batch shape [N]). The predictive distributions.
    :param true: torch.Tensor [N]. The (real) predictions.
    :param confidence: float. The confidence interval in the range (0, 1).
    :return: calibration score (ideally should be close to confidence)
    """
    if isinstance(pred, torch.distributions.distribution.Distribution):
        cdf = pred.cdf(true)
        pvals = 2 * torch.min(cdf, 1 - cdf)
        return (pvals <= confidence).float().mean()
    else: # cdf method is nor implemented in PyTorch for StudentT
        cdf = scipy.stats.t.cdf(true.cpu(), df=pred["df"].cpu(), loc=pred["loc"].cpu(), scale=pred["scale"].cpu())
        cdf = torch.from_numpy(cdf)
        pvals = 2 * torch.min(cdf, 1 - cdf)
        return (pvals <= confidence).float().mean()


def calibration_regression(pred, true):
    combined = 0
    for p in np.arange(0.1, 1, 0.1):
        combined += (_calibration_regression(pred, true, p) - p) ** 2
    return sqrt(combined)

def nll(pred, true):
    """
    :param pred:  torch.distributions (batch shape [N]). The predictive distributions.
    :param true: torch.Tensor [N]. The (real) predictions.
    :return: real. negative log-likelihood
    """
    return - pred.log_prob(true).float().mean()
