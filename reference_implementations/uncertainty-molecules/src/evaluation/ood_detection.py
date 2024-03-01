import numpy as np
import torch
from sklearn import metrics


def auc_roc(pred, true):
    fpr, tpr, thresholds = metrics.roc_curve(pred.reshape(-1), true.reshape(-1))
    return metrics.auc(fpr, tpr)

def auc_apr(pred, true):
    return metrics.average_precision_score(pred.reshape(-1), true.reshape(-1))

# OOD detection metrics
# check pytorch metrics for speedup
def anomaly_detection(sigmas, ood_sigmas, score_type='AUROC', uncertainty_type='aleatoric'):
    
    corrects = np.concatenate([np.ones(sigmas.size(0)), np.zeros(ood_sigmas.size(0))], axis=0) # 1 for ID class, 0 for OOD
    scores = np.concatenate([sigmas, ood_sigmas], axis=0)
        
    if score_type == 'AUROC':
        return auc_roc(corrects, scores)
    elif score_type == 'APR':
        return auc_apr(corrects, scores)
    else:
        raise NotImplementedError
    
    
    
    
