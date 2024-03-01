import torch
import numpy as np
import scipy.sparse as sp
import scipy
import itertools
import math
from tqdm.auto import tqdm


def accuracy(y, y_hat, abstain, idx):
    y = y[~abstain]
    y_hat = y_hat[~abstain]
    return (y == y_hat).sum()/idx.shape[0]


def certified_accuracy(y, y_hat, robust_not_abstained, idx):
    return (y[robust_not_abstained] == y_hat[robust_not_abstained]).sum()/idx.shape[0]


def average_results(exp, key, seeds):
    max_rd = max([exp[k][key].shape[0] for k in seeds])
    max_ra = max([exp[k][key].shape[1] for k in seeds])
    merged = np.zeros((len(seeds), max_rd, max_ra))

    for i in range(len(seeds)):
        array_i = exp[seeds[i]][key]
        merged[i, :array_i.shape[0], :array_i.shape[1]] += array_i

    return minimize(merged.mean(0).T), minimize(merged.std(0).T)


def minimize(array: np.array):
    return array.shape, tuple(t.tolist() for t in array.nonzero()), array[array.nonzero()].tolist()
