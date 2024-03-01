import numpy as np
import logging
import torch

import torch_geometric.transforms as T

from scipy.stats import entropy
from sklearn.metrics import auc

from equivariance_robustness.sparse_smoothing.utils import *
from equivariance_robustness.sparse_smoothing.prediction import *
from equivariance_robustness.sparse_smoothing.training import *
from equivariance_robustness.data import *
from equivariance_robustness.sparse_smoothing.models import *

from sparse_smoothing.cert import binary_certificate, joint_binary_certificate


class Experiment():

    def run(self, hparams):
        result_keys = ["clean_acc", "abstain", "cert_acc",
                       "cert_acc_curve", "cert_ratio", "cert_ratio_curve"]

        results = {}
        dict_to_save = {}
        nc = hparams['out_channels']

        for seed in range(5):
            dict_to_save[seed] = {}

            model = create_model(hparams, seed)

            if hparams["task"] == "node_classification":
                data = load_node_classification_dataset(hparams["dataset"],
                                                        hparams["dataset_path"],
                                                        seed)
                idx_train = data.train_mask
                idx_valid = data.val_mask
                idx_test = data.test_mask
                model = training_node_classification(model,
                                                     data,
                                                     idx_train,
                                                     idx_valid,
                                                     hparams).eval()
                pre_votes = smoothed_gnn(hparams, model, data, idx_test, nc,
                                         hparams["n0"], batch_size=hparams["batch_size"])
                votes = smoothed_gnn(hparams, model, data, idx_test, nc,
                                     hparams["n1"], batch_size=hparams["batch_size"])

                y_hat = pre_votes.cpu().argmax(1).numpy()
                correct = (data.y[idx_test].cpu().numpy() == y_hat)
                print(f"acc={correct.mean()}")
            else:
                dataloader = load_graph_classification_dataset(hparams["dataset"],
                                                               hparams["dataset_path"],
                                                               seed)
                model = training_graph_classification(model,
                                                      dataloader[0],
                                                      dataloader[1],
                                                      hparams).eval()
                pre_votes = smoothed_gnn_graph_classification(hparams, model,
                                                              dataloader[2], nc,
                                                              hparams["n0"])
                votes = smoothed_gnn_graph_classification(hparams, model,
                                                          dataloader[2], nc,
                                                          hparams["n1"])

                y_hat = pre_votes.argmax(1).cpu().numpy()
                y = np.array([data.y.item() for data in dataloader[2]])
                correct = (y == y_hat)
                print(f"acc={correct.mean()}")

            # compute certificates
            pf_plus_adj = hparams['p_adj_plus']
            pf_plus_att = hparams['p_att_plus']
            pf_minus_adj = hparams['p_adj_minus']
            pf_minus_att = hparams['p_att_minus']
            if pf_plus_adj == 0 and pf_minus_adj == 0:
                print('Just ATT')
                grid_base, grid_lower, grid_upper = binary_certificate(
                    votes=votes, pre_votes=pre_votes, n_samples=hparams["n1"],
                    conf_alpha=hparams["alpha"],
                    pf_plus=pf_plus_att, pf_minus=pf_minus_att)
            # we are perturbing ONLY the GRAPH
            elif pf_plus_att == 0 and pf_minus_att == 0:
                print('Just ADJ')
                grid_base, grid_lower, grid_upper = binary_certificate(
                    votes=votes, pre_votes=pre_votes, n_samples=hparams["n1"],
                    conf_alpha=hparams["alpha"],
                    pf_plus=pf_plus_adj, pf_minus=pf_minus_adj)
            else:
                grid_base, grid_lower, grid_upper = joint_binary_certificate(
                    votes=votes, pre_votes=pre_votes, n_samples=hparams["n1"],
                    conf_alpha=hparams["alpha"],
                    pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
                    pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att)

            dict_to_save[seed]['pre_votes'] = pre_votes
            dict_to_save[seed]['votes'] = votes
            dict_to_save[seed]['grid_base'] = grid_base
            dict_to_save[seed]['grid_lower'] = grid_lower
            dict_to_save[seed]['grid_upper'] = grid_upper
            dict_to_save[seed]['y_hat'] = y_hat

            binary_class_cert = grid_base > 0.5
            multi_class_cert = grid_lower >= grid_upper

            dict_to_save[seed]['binary_class_cert'] = binary_class_cert
            dict_to_save[seed]['multi_class_cert'] = multi_class_cert

            binary_class_cert_ratios = binary_class_cert.mean(0)
            binary_class_cert_acc = (correct * binary_class_cert.T).T.mean(0)
            multi_class_cert_ratios = multi_class_cert.mean(0)
            multi_class_cert_acc = (correct * multi_class_cert.T).T.mean(0)

            dict_to_save[seed]['clean_acc'] = correct.mean()
            dict_to_save[seed]['binary_class_cert_ratios'] = binary_class_cert_ratios
            dict_to_save[seed]['binary_class_cert_acc'] = binary_class_cert_acc
            dict_to_save[seed]['multi_class_cert_ratios'] = multi_class_cert_ratios
            dict_to_save[seed]['multi_class_cert_acc'] = multi_class_cert_acc

        dict_to_save["p_att_plus"] = hparams["p_att_plus"]
        dict_to_save["p_att_minus"] = hparams["p_att_minus"]
        dict_to_save["p_adj_plus"] = hparams["p_adj_plus"]
        dict_to_save["p_adj_minus"] = hparams["p_adj_minus"]
        return results, dict_to_save
