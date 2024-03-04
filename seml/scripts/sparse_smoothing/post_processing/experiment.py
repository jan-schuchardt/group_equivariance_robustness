import numpy as np
import logging
import torch

import seml
from equivariance_robustness.data import *

from tqdm.auto import tqdm


class Experiment():

    def run(self, hparams):

        df_results = seml.get_results('graph_cert_sparse_smoothing', to_data_frame=True)

        id = hparams['exp']
        load_dir = df_results.iloc[id]["config.conf.save_dir"] + "/" + \
            df_results.iloc[id]["config.db_collection"] + "_"
        save_dir = hparams['save_dir'] + \
            df_results.iloc[id]["config.db_collection"] + "_"
        dataset_name = df_results.iloc[id]["config.hparams.dataset"]
        dataset_path = df_results.iloc[id]["config.hparams.dataset_path"]

        path = load_dir + str(df_results.iloc[id]["config.overwrite"])
        data = torch.load(path)

        correct = {}
        for seed in range(5):
            if df_results.iloc[id]['config.hparams.task'] == 'node_classification':
                dataset = load_node_classification_dataset(name=dataset_name,
                                                           root=dataset_path,
                                                           seed=seed,
                                                           device="cpu")
                idx_test = dataset.test_mask
                correct[seed] = (data[seed]['y_hat'] ==
                                 dataset.y[idx_test].numpy())
            else:
                dataloader = load_graph_classification_dataset(dataset_name,
                                                               dataset_path,
                                                               seed=seed)
                y = np.array([data.y.item() for data in dataloader[2]])
                correct[seed] = (data[seed]['y_hat'] == y)

        cert_ratio = {}
        cert_acc = {}

        for cert_name in tqdm(['binary_class_cert', 'multi_class_cert']):
            if (df_results.iloc[id]['config.hparams.p_adj_plus'] > 0) &\
                (df_results.iloc[id]['config.hparams.p_adj_minus'] > 0) &\
                (df_results.iloc[id]['config.hparams.p_att_plus'] > 0) &\
                    (df_results.iloc[id]['config.hparams.p_att_minus'] > 0):
                # ca_A, cd_A, ca_F, cd_F, i.e. joint cert
                for i_ca_A in tqdm(np.arange(1, 10, 1)):
                    for i_cd_A in np.arange(1, 10, 1):
                        for i_ca_F in np.arange(1, 10, 1):
                            for i_cd_F in np.arange(1, 10, 1):
                                cert_ratio[(cert_name, i_ca_A, i_cd_A, i_ca_F, i_cd_F)] = self.compute_ged_joint(
                                    data, i_ca_A, i_cd_A, i_ca_F, i_cd_F, cert_name, correct, cert_acc=False)
                                cert_acc[(cert_name, i_ca_A, i_cd_A, i_ca_F, i_cd_F)] = self.compute_ged_joint(
                                    data, i_ca_A, i_cd_A, i_ca_F, i_cd_F, cert_name, correct, cert_acc=True)
            else:
                for i_ca in np.arange(1, 10, 1):
                    for i_cd in np.arange(1, 10, 1):
                        if i_ca == i_cd and i_ca != 1:
                            continue
                        cert_ratio[(cert_name, i_ca, i_cd)] = self.compute_ged(
                            data, i_ca, i_cd, cert_name, correct, cert_acc=False)
                        cert_acc[(cert_name, i_ca, i_cd)] = self.compute_ged(
                            data, i_ca, i_cd, cert_name, correct, cert_acc=True)

        path = save_dir + str(df_results.iloc[id]["config.overwrite"])
        result = {'cert_ratio': cert_ratio, 'cert_acc': cert_acc}
        data = torch.save(result, path)
        return {}, {}

    def post_process(self, x, y, z):
        x_new = []
        y_new = []
        z_new = []
        for i in range(y.shape[0]):
            if i == 0 or y[i-1] != y[i] or i == y.shape[0]-1:
                x_new.append(x[i])
                y_new.append(y[i])
                z_new.append(z[i])
        return x_new, y_new, z_new

    def compute_ged(self, data, ca, cd, cert_name, correct, cert_acc=False, max_radius=30):
        x = np.arange(0, max_radius+0.1, 0.1)
        cert_ratios = []
        for seed in np.arange(0, 5, 1):
            cert_ratios_seed = []
            for r in x:
                cert = data[seed][cert_name]
                ged = self.compute_ged_single(cert, ca, cd, r)
                if cert_acc:
                    ged = ged * correct[seed]
                cert_ratios_seed.append(ged.mean(0))
            cert_ratios.append(cert_ratios_seed)
        cert_ratios = np.array(cert_ratios)
        return self.post_process(x, cert_ratios.mean(0), cert_ratios.std(0))

    def compute_ged_joint(self, data, ca_A, cd_A, ca_F, cd_F, cert_name, correct, cert_acc=False, max_radius=30):
        x = np.arange(0, max_radius+0.1, 0.1)
        cert_ratios = []
        for seed in np.arange(0, 5, 1):
            cert_ratios_seed = []
            for r in x:
                cert = data[seed][cert_name]
                ged = self.compute_ged_multiple(
                    cert, ca_A, cd_A, ca_F, cd_F, r)
                if cert_acc:
                    ged = ged * correct[seed]
                cert_ratios_seed.append(ged.mean(0))
            cert_ratios.append(cert_ratios_seed)
        cert_ratios = np.array(cert_ratios)
        return self.post_process(x, cert_ratios.mean(0), cert_ratios.std(0))

    def compute_ged_single(self, base_cert, ca, cd, r):
        """
            base_cert.shape = (nodes, r_a, r_d)
        """
        result = []
        max_ra = np.floor(np.round(r/ca, 10)).astype(int)
        # cert = np.zeros(shape=(base_cert.shape[0], max_ra+1))
        for ra in range(0, max_ra+1):
            rd = np.floor(np.round((r-ca*ra)/cd, 10)).astype(int)
            # print(ra, rd, ra*ca + rd*cd, r)
            if ra < base_cert.shape[1] and rd < base_cert.shape[2] and ra >= 0 and rd >= 0:
                result.append(base_cert[:, ra, rd])
            else:
                result.append([False for i in range(base_cert.shape[0])])
        return np.array(result).all(axis=0)

    def compute_ged_multiple(self, base_cert, ca_A, cd_A, ca_F, cd_F, r):
        """
            base_cert.shape = (nodes, ra_A, rd_A, ra_F, rd_F)
        """
        result = []
        max_ra_A = np.floor(np.round(r/ca_A, 10)).astype(int)
        for ra_A in range(0, max_ra_A+1):
            max_rd_A = np.floor(np.round((r-ca_A*ra_A)/cd_A, 10)).astype(int)
            for rd_A in range(0, max_rd_A+1):
                max_ra_F = np.floor(
                    np.round((r-ca_A*ra_A-cd_A*rd_A)/ca_F, 10)).astype(int)
                for ra_F in range(0, max_ra_F+1):
                    rd_F = np.floor(
                        np.round((r-ca_A*ra_A-cd_A*rd_A-ca_F*ra_F)/cd_F, 10)).astype(int)

                    if ra_A < base_cert.shape[1] and \
                        rd_A < base_cert.shape[2] and \
                            ra_F < base_cert.shape[3] and \
                        rd_F < base_cert.shape[4]:
                        result.append(base_cert[:, ra_A, rd_A, ra_F, rd_F])
                    else:
                        result.append(
                            [False for i in range(base_cert.shape[0])])
        return np.array(result).all(axis=0)
