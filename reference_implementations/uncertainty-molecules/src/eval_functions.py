from pkgutil import get_data
from src.datasets.data_provider import get_dataloader
from torch_geometric.loader import DataLoader
import torch
from src.metrics.ood_detection import anomaly_detection 

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def eval(model, ood_datasets, id_loader, target, batch_size, encoding_dim):
    metrics = {}
    for ood_params in ood_datasets:
        _, _, ood_loader, _ = get_dataloader(
            **ood_params, 
            target=target, 
            batch_size=batch_size,
            dimension=encoding_dim
        )
        ood_set_name = ood_params['dataset_name'] + '_' + str(ood_params['transform_params'])
        metrics[ood_set_name] = test_ood(model, id_loader, ood_loader, ood_set_name)
    return metrics
            

def test_ood(model, loader, ood_loader, ood_ds_name):
    model.eval()
    metrics = {}
    with torch.no_grad():
        stddevs = eval_dataset(model, loader, False)
        ood_stddevs = eval_dataset(model, ood_loader)
            
        metrics[f'AUROC_{ood_ds_name}'] = anomaly_detection(1/stddevs, 1/ood_stddevs, score_type='AUROC')
        # apr in and apr out
        metrics[f'APR_{ood_ds_name}'] = anomaly_detection(1/stddevs, 1/ood_stddevs, score_type='APR')
        metrics[f'APR_IN_{ood_ds_name}'] = anomaly_detection(1/ood_stddevs, 1/stddevs, score_type='APR')
        metrics[f'stddevs_{ood_ds_name}'] = ood_stddevs.mean()
        metrics[f'val_stddevs_in_{ood_ds_name}'] = stddevs.mean()
        #if True or (ood_stddevs > stddevs).any():
        if False:
            print(f'stddevs: {stddevs.mean()}, ood_stddevs: {ood_stddevs.mean()}')
            print(anomaly_detection(1/stddevs, 1/ood_stddevs, score_type='AUROC'))
            print(f'stddevs: {stddevs.shape}, ood_stddevs: {ood_stddevs.shape}')
            
        return metrics

def eval_dataset(model, loader, isval=False, get_preds=False):
    stddevs = []
    predictions = []
    model = model.to(device)
    if isval:
        print(' OOD batches')
    for batch in loader:
        batch = batch.to(device)
        # TODO: change here to get the force variance potentially
        #x = (batch.z, batch.pos, batch.batch)
        preds = model(batch)
        if isinstance(preds, tuple):
            energy, forces = preds
        else:
            energy = preds
        #stddevs.append(preds.stddev.cpu().reshape(-1))
        predictions.append(preds.mean.cpu().reshape(-1))
        stddevs.append(energy.stddev.detach().cpu().reshape(-1))
    if get_preds:
        return predictions, stddevs
    return torch.cat(stddevs)