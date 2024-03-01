import numpy as np
import logging
import torch
import seml
import random
import numpy as np
import scipy.sparse as sp

from equivariance_robustness.piPPNP.cert import *
from graph_cert.utils import *
from equivariance_robustness.data import load_node_classification_dataset
from graph_cert.models import PiPPNP
from equivariance_robustness.piPPNP.training import train


class Experiment():

    def run(self, hparams):

        ca = hparams['ca']
        cd = hparams['cd']
        threat_model = hparams['threat_model']  # rem or add_rem
        dataset = hparams['dataset']
        alpha = hparams['alpha']
        n_hidden = hparams['n_hidden']

        dict_to_save = {}
        dict_to_save["local"] = {}

        for seed in range(5):
            data = load_node_classification_dataset(name=dataset,
                                                    root=hparams["dataset_path"],
                                                    seed=seed, device=hparams['device'])
            idx_train = data.train_mask.cpu()
            idx_valid = data.val_mask.cpu()
            idx_test = data.test_mask.cpu()

            # prepare data
            edge_idx = data.edge_index.cpu().numpy()
            edge_data = (np.ones(data.edge_index.shape[1]),
                         (edge_idx[0], edge_idx[1]))
            adj = sp.csr_matrix(edge_data,
                                shape=(data.x.shape[0], data.x.shape[0]))
            X = data.x.cpu()
            labels = data.y.cpu()
            deg = adj.sum(1).A1.astype(np.int32)

            # PageRank Matrix of the clean graph
            ppr_clean = propagation_matrix(adj=adj, alpha=alpha)
            ppr_clean = torch.tensor(ppr_clean, dtype=torch.float32)

            # train model
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            model = PiPPNP(X.shape[1], labels.max().item()+1,
                           n_hidden=[n_hidden])
            _ = train(model, X, ppr_clean, labels, idx_train, idx_valid,
                      hparams['lr'], hparams['weight_decay'], hparams['patience'], hparams['max_epochs'])
            model.eval()

            # final forward-pass
            logits, _ = model(X, ppr_clean)
            logits = logits.detach().numpy()
            weighted_logits = ppr_clean @ logits
            predicted = weighted_logits.argmax(1)
            correct = (labels[idx_test] == predicted[idx_test]).float()

            # compute threat model
            fragile = get_fragile(adj=adj, threat_model=threat_model)

            ratios_certifed = []
            accs_certified = []

            for local_strength in tqdm(np.arange(1, 10+1)):
                # set the local budget proportional to the node degree
                local_budget = np.maximum(deg - 11 + local_strength, 0)
                # print(local_budget.max())

                # precomputed the K x K perturbed graphs
                k_squared_pageranks = k_squared_parallel(
                    adj=adj, alpha=alpha, fragile=fragile, local_budget=local_budget, ca=ca, cd=cd,
                    threat_model=threat_model, logits=logits, nodes=np.arange(adj.shape[0])[idx_test])

                # compute the exact worst-case margins for all test nodes
                worst_margins = worst_margins_given_k_squared(
                    k_squared_pageranks=k_squared_pageranks, labels=predicted[idx_test], logits=logits)

                ratios_certifed.append((worst_margins > 0).mean())
                accs_certified.append(
                    ((worst_margins > 0)*correct.numpy()).mean())

            dict_to_save["local"][seed] = {}
            dict_to_save["local"][seed]["cert_ratios"] = ratios_certifed
            dict_to_save["local"][seed]["cert_accs"] = accs_certified

        ratios = [dict_to_save["local"][seed]["cert_ratios"]
                  for seed in range(1)]
        accs = [dict_to_save["local"][seed]["cert_accs"]
                for seed in range(1)]

        dict_to_save["local"]["cert_ratios"] = np.mean(
            ratios, axis=0), np.std(ratios, axis=0)
        dict_to_save["local"]["cert_accs"] = np.mean(
            accs, axis=0), np.std(accs, axis=0)

        return {}, dict_to_save
