import torch
import numpy as np
from sparse_smoothing.utils import sample_multiple_graphs


def sample_graphs(data, hparams, batch_size):
    sample_config = {}
    sample_config['pf_plus_adj'] = hparams['p_adj_plus']
    sample_config['pf_plus_att'] = hparams['p_att_plus']
    sample_config['pf_minus_adj'] = hparams['p_adj_minus']
    sample_config['pf_minus_att'] = hparams['p_att_minus']

    n, d = data.x.shape
    attr_idx = data.x.nonzero().T
    per_attr_idx, per_edge_idx = sample_multiple_graphs(attr_idx, data.edge_index,
                                                        sample_config, n, d,
                                                        nsamples=batch_size)
    per_x = torch.zeros((batch_size*n, d),
                        device=data.x.device)
    per_x[per_attr_idx[0], per_attr_idx[1]] = 1
    return per_x, per_edge_idx
