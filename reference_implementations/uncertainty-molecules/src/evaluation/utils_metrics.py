import torch
from torchmetrics import AUROC
from src.evaluation.aucpr import AUCPR
from src.evaluation.calibration_scores import calibration_regression
from tqdm import tqdm


def get_metrics_and_uncertainties_dropout(model, data_loader):
    energy_maes = []
    forces_maes = []
    calibrations = []
    
    energy_uncertainties = []
    dets = []
    traces = []
    largest_eigs = []
    max_vals, min_vals, mean_vals = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in tqdm(data_loader):
        data = data.to(device)
        output_dict = model(data)
        energy, forces, energy_uncertainty, det, trace, largest_eig, energy_distribution = output_dict['energy'], output_dict['forces'], output_dict['energy_uncertainty'], output_dict['det'], output_dict['trace'], output_dict['largest_eig'], output_dict['energy_distribution']
        #max_val, min_val, mean_val = output_dict['max_val'], output_dict['min_val'], output_dict['mean_val']
        with torch.no_grad():
            energy = energy.view(-1)
            energy_mae = (energy - data.energy).abs().mean()
            forces_mae = (forces - data.force).abs().mean()
            calibration = calibration_regression(energy_distribution, data.energy)

            energy_maes.append(energy_mae.detach().cpu())
            forces_maes.append(forces_mae.detach().cpu())
            calibrations.append(calibration)
            
            energy_uncertainties.append(energy_uncertainty.detach().cpu())
            dets.append(det.detach().cpu())
            traces.append(trace.detach().cpu())
            largest_eigs.append(largest_eig.detach().cpu())
            
            # max_vals.append(max_val.detach().cpu())
            # min_vals.append(min_val.detach().cpu())
            # mean_vals.append(mean_val.detach().cpu())

    energy_mae = torch.FloatTensor(energy_maes)
    forces_mae = torch.FloatTensor(forces_maes)
    calibrations = torch.FloatTensor(calibrations)
    
    return energy_mae.mean(), forces_mae.mean(), calibrations.mean(), torch.stack(energy_uncertainties), torch.stack(dets), torch.stack(traces), torch.stack(largest_eigs)#, torch.stack(max_vals), torch.stack(min_vals), torch.stack(mean_vals)

def get_maes_and_calibration_evidential(model, data_loader):
    energy_maes = []
    forces_maes = []
    calibrations = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in data_loader:
        data = data.to(device)
        output_dict = model(data)
        energy, forces, energy_distribution = output_dict['energy'], output_dict['forces'], output_dict['energy_distribution']
        with torch.no_grad():
            energy = energy.view(-1)
            energy_mae = (energy - data.energy).abs().mean()
            forces_mae = (forces - data.force).abs().mean()
            calibration = calibration_regression(energy_distribution, data.energy)

            energy_maes.append(energy_mae)
            forces_maes.append(forces_mae)
            calibrations.append(calibration)

    energy_mae = torch.FloatTensor(energy_maes)
    forces_mae = torch.FloatTensor(forces_maes)
    calibrations = torch.FloatTensor(calibrations)
    return energy_mae.mean(), forces_mae.mean(), calibrations.mean()

def get_uncertainties_evidential(model, data_loader):
    uncertainties_1 = []
    uncertainties_2 = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in data_loader:
        data = data.to(device)
        output_dict = model(data)
        uncertainty_1, uncertainty_2 = output_dict['uncertainty_1'], output_dict['uncertainty_2']
        uncertainties_1.append(uncertainty_1.detach().cpu())
        uncertainties_2.append(uncertainty_2.detach().cpu())
    return torch.cat(uncertainties_1), torch.cat(uncertainties_2)


def get_uncertainty_metrics(model, combined_loader):
    roc1 = AUROC(pos_label=1)
    pr1 = AUCPR()
    roc2 = AUROC(pos_label=1)
    pr2 = AUCPR()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch in combined_loader:
        data, y = batch
        data = data.to(device)
        y = y.to(device)

        output_dict = model(data)
        uncertainty_1, uncertainty_2 = output_dict['uncertainty_1'], output_dict['uncertainty_2']

        roc1.update(-uncertainty_1, y)
        pr1.update(-uncertainty_1, y)
        roc2.update(-uncertainty_2, y)
        pr2.update(-uncertainty_2, y)

    roc1 = roc1.compute()
    pr1 = pr1.compute()
    roc2 = roc2.compute()
    pr2 = pr2.compute()
    return roc1, pr1, roc2, pr2

    
def run_predictions_for_timer(model, data_loader):
    outputs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in data_loader:
        data = data.to(device)
        output_dict = model(data)
        outputs.append(output_dict)
    return outputs