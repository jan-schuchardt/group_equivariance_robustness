import torch
from torch_scatter import scatter
import torch.distributions as D
from torch.autograd import grad
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UQModel(torch.nn.Module):
    """
    A wrapper class for an Uncertainty-Quantification Model
    Args:
        - model_name: "dropout_dimenet++" or "evidential_dimenet++"
        - model: trained model
    """
    def __init__(self, model_name, model, n_mc_dropout_runs=69):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.n_mc_dropout_runs = n_mc_dropout_runs
        
    def forward(self, inputs):
        """
        Args:
            inputs: input batch containing inputs.z, inputs,pos, inputs.batch
        Returns:
            Dict consisting of:
                - energy: energy prediction
                - forces: forces prediction
                - uncertainty_1: energy std for dropout model / epistemic for evidential model
                - uncertainty_2: forces cov det for dropout model / aleatoric for evidential model
        """
        if self.model_name == "dropout":
            energy_samples, forces_samples = self.predict_MC_Dropout(inputs, n_mc_dropout_runs=self.n_mc_dropout_runs)
            with torch.no_grad():
                energy, forces = energy_samples.mean(0).to(device), forces_samples.mean(0).to(device)
                energy_uncertainty = energy_samples.std(0).squeeze().to(device)
                det, trace, largest_eig = self.compute_forces_uncertainty_molecule(forces_samples, inputs)
                                
                energy_distribution = D.normal.Normal(energy, energy_uncertainty)
                return {"energy": energy, "forces": forces, "energy_uncertainty": energy_uncertainty, "det": det.to(device), "trace": trace.to(device), "largest_eig": largest_eig.to(device), "energy_distribution": energy_distribution}
        
        elif self.model_name == "evidential_dimenet++":
            energy, forces = self.model(inputs.z, inputs.pos, inputs.batch)
            with torch.no_grad():
                gamma, v, alpha, beta = torch.split(energy, 1, dim=-1)
                gamma, v, alpha, beta = gamma.view(-1), v.view(-1), alpha.view(-1), beta.view(-1)
                epistemic = beta/(v*(alpha-1))
                aleatoric = beta/(alpha-1)
                energy_distribution = {"df":2*alpha, "loc":gamma, "scale":beta*(1+v)/(v*alpha)}
                return {"energy": gamma, "forces": forces, "uncertainty_1": epistemic, "uncertainty_2": aleatoric, "energy_distribution": energy_distribution}

            
    def predict_MC_Dropout(self, inputs, n_mc_dropout_runs=69):
        energy_samples = []
        forces_samples = []
        for _ in range(n_mc_dropout_runs):
            energy, forces = self.model(inputs)
            energy_samples.append(energy)
            forces_samples.append(forces)
            
        energy_samples = torch.stack(energy_samples).detach().cpu()
        forces_samples = torch.stack(forces_samples).detach().cpu()
        return (energy_samples, forces_samples)
    
    def compute_forces_uncertainty_atomwise(self, forces_samples, inputs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        det = torch.zeros(forces_samples.shape[1], device=device) # Number of atoms
        trace = torch.zeros(forces_samples.shape[1], device=device)
        largest_eig = torch.zeros(forces_samples.shape[1], device=device)
        for atom_idx in range(forces_samples.shape[1]):
            atom_force = forces_samples[:, atom_idx, :]
            atom_cov = torch.tensor(np.cov(atom_force.double().T.cpu().numpy())).to(device)
            det[atom_idx] = torch.det(atom_cov)
            trace[atom_idx] = torch.trace(atom_cov)
            eig_vals = torch.linalg.eigvalsh(atom_cov)
            largest_eig[atom_idx] = eig_vals[-1]

        det = scatter(det, inputs.batch, dim=0, reduce="mean") # Average over atoms for each molecule
        trace = scatter(trace, inputs.batch, dim=0, reduce="mean")
        largest_eig = scatter(largest_eig, inputs.batch, dim=0, reduce="mean")
        return det, trace, largest_eig
    
    def compute_forces_uncertainty_molecule(self, forces_samples, inputs):
        # forces_samples has shape (n_mc_dropout_runs, N_atoms, 3) -> we make it (n_mc_dropout_runs, 3*N_atoms)
        forces_samples = torch.reshape(forces_samples, (forces_samples.shape[0], -1))
        cov_mat = torch.tensor(np.cov(forces_samples.double().T.cpu().numpy())).to(device)
        det = torch.det(cov_mat)
        trace = torch.trace(cov_mat)
        eig_vals = torch.linalg.eigvalsh(cov_mat)
        largest_eig = eig_vals[-1]

        return det, trace, largest_eig
    
    
class Ensemble(torch.nn.Module):
    """
    A wrapper class for an Uncertainty-Quantification Model
    Args:
        - model_name: "dropout_dimenet++" or "evidential_dimenet++"
        - model: trained model
    """
    def __init__(self, model_name, models):
        super().__init__()
        self.model_name = model_name
        self.models = models
        self.n_models = len(models)
        
    def forward(self, inputs):
        """
        Args:
            inputs: input batch containing inputs.z, inputs,pos, inputs.batch
        Returns:
            Dict consisting of:
                - energy: energy prediction
                - forces: forces prediction
                - uncertainty_1: energy std for dropout model / epistemic for evidential model
                - uncertainty_2: forces cov det for dropout model / aleatoric for evidential model
        """
        energy_samples, forces_samples = self.predict(inputs)
        with torch.no_grad():
            energy, forces = energy_samples.mean(0).to(device), forces_samples.mean(0).to(device)
            energy_uncertainty = energy_samples.std(0).squeeze().to(device)
            det, trace, largest_eig = self.compute_forces_uncertainty_molecule(forces_samples, inputs)
            
            try:             
                energy_distribution = D.normal.Normal(energy, energy_uncertainty + 10e-6)
            except:
                print('not working')
                energy_distribution = D.normal.Normal(energy, energy_uncertainty)
            return {
                "energy": energy, 
                "forces": forces, 
                "energy_uncertainty": energy_uncertainty, 
                "det": det.to(device), 
                "trace": trace.to(device), 
                "largest_eig": largest_eig.to(device), 
                "energy_distribution": energy_distribution
            }
            
    def predict(self, inputs):
        energy_samples = []
        forces_samples = []
        orig_data = inputs.clone()
        for model in self.models:
            inputs = orig_data.clone()
            inputs.pos.requires_grad = True
            energy = model(inputs)
            forces = - grad(
                outputs=energy.sum(),
                inputs=inputs.pos,
                create_graph=True,
                retain_graph=True
            )[0]
            energy_samples.append(energy)
            forces_samples.append(forces)
            inputs.pos.requires_grad = False
            
        energy_samples = torch.stack(energy_samples).detach().cpu()
        forces_samples = torch.stack(forces_samples).detach().cpu()
        return (energy_samples, forces_samples)
    
    def compute_forces_uncertainty_atomwise(self, forces_samples, inputs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        det = torch.zeros(forces_samples.shape[1], device=device) # Number of atoms
        trace = torch.zeros(forces_samples.shape[1], device=device)
        largest_eig = torch.zeros(forces_samples.shape[1], device=device)
        for atom_idx in range(forces_samples.shape[1]):
            atom_force = forces_samples[:, atom_idx, :]
            atom_cov = torch.tensor(np.cov(atom_force.double().T.cpu().numpy())).to(device)
            det[atom_idx] = torch.det(atom_cov)
            trace[atom_idx] = torch.trace(atom_cov)
            eig_vals = torch.linalg.eigvalsh(atom_cov)
            largest_eig[atom_idx] = eig_vals[-1]

        det = scatter(det, inputs.batch, dim=0, reduce="mean") # Average over atoms for each molecule
        trace = scatter(trace, inputs.batch, dim=0, reduce="mean")
        largest_eig = scatter(largest_eig, inputs.batch, dim=0, reduce="mean")
        return det, trace, largest_eig
    
    def compute_forces_uncertainty_molecule(self, forces_samples, inputs):
        # forces_samples has shape (n_mc_dropout_runs, N_atoms, 3) -> we make it (n_mc_dropout_runs, 3*N_atoms)
        forces_samples = torch.reshape(forces_samples, (forces_samples.shape[0], -1))
        cov_mat = torch.tensor(np.cov(forces_samples.double().T.cpu().numpy())).to(device)
        det = torch.det(cov_mat)
        trace = torch.trace(cov_mat)
        eig_vals = torch.linalg.eigvalsh(cov_mat)
        largest_eig = eig_vals[-1]

        return det, trace, largest_eig
    
    def eval(self):
        for model in self.models:
            model.eval()
        return self

    def train(self):
        for model in self.models:
            model.train()
        return self

class GPModel(torch.nn.Module):
    """
    A wrapper class for an Uncertainty-Quantification Model
    Args:
        - model_name: "dropout_dimenet++" or "evidential_dimenet++"
        - model: trained model
    """
    def __init__(self, model_name, feature_extractor, gp, likelihood, epsilon):
        super().__init__()
        self.model_name = model_name
        self.fe = feature_extractor.to(device).eval()
        self.gp = gp.to(device).eval()
        self.likelihood = likelihood.to(device).eval()
        self.epsilon = epsilon

        
    def forward(self, inputs):
        """
        Args:
            inputs: input batch containing inputs.z, inputs,pos, inputs.batch
        Returns:
            Dict consisting of:
                - energy: energy prediction
                - forces: forces prediction
                - uncertainty_1: energy std for dropout model / epistemic for evidential model
                - uncertainty_2: forces cov det for dropout model / aleatoric for evidential model
        """
        assert max(inputs.batch) == 0, "Only BS 1 implemented"
        # create epsilon environment
        unchanged_batch = inputs.clone()
        n_atoms = len(inputs.pos)
        positions = [inputs.pos]
        inputs.pos.requires_grad = True
        for i in range(n_atoms * 3):
            current_vec = torch.zeros(n_atoms * 3).cuda()
            current_vec[i] = self.epsilon
            positions.append(inputs.pos + current_vec.reshape(n_atoms, 3))

        embeddings = []
        for p in positions:
            current_batch = inputs.clone()
            current_batch.pos = p
            embeddings.append(self.fe(current_batch))
        eps_env_embeddings = torch.cat(embeddings)
        energy_dist = self.gp(eps_env_embeddings)
        
        if self.model_name == "mult-gp":
            energy, covar = self.aggregate_dist(energy_dist, n_atoms)
        else:
            energy, covar = energy_dist.loc, energy_dist.covariance_matrix
        
        forces = - grad(
                    outputs=energy[0],
                    inputs=inputs.pos,
                    create_graph=False,
                    retain_graph=False
                )[0]
        with torch.no_grad():
            #det, trace, largest_eig, max_val, min_val, mean_val = self.compute_forces_uncertainty_molecule(energy_dist, inputs)
            det, trace, largest_eig = self.compute_forces_uncertainty_molecule(covar, inputs)
            
            #pred_energy = energy[0]
            #energy_uncertainty = covar[0,0]
            energy_distribution = self.gp(self.fe(unchanged_batch), batch=unchanged_batch.batch)
            pred_energy = energy_distribution.loc
            energy_uncertainty = energy_distribution.variance
            try:               
                energy_distribution = D.normal.Normal(pred_energy, energy_uncertainty + 10e-6)
            except:
                print(pred_energy)
                print(energy_uncertainty)
                energy_distribution = D.normal.Normal(pred_energy, torch.tensor([0.1]).to(pred_energy))
            return {
                "energy": pred_energy, 
                "forces": forces, 
                "energy_uncertainty": energy_uncertainty, 
                "det": det.to(device), 
                "trace": trace.to(device), 
                "largest_eig": largest_eig.to(device), 
                "energy_distribution": energy_distribution,
                # "max_val": max_val, 
                # "min_val": min_val,
                # "mean_val": mean_val
            }
            
    def aggregate_dist(self, energy_dist, n_atoms, fast_approx=False):
        operator = []
        if fast_approx:
            num_ops = 4
        else:
            num_ops = 3 * n_atoms + 1
        for i in range(num_ops):
            row = torch.zeros(1, ((num_ops) * n_atoms))
            row[:, i*n_atoms:n_atoms*(i+1)] += 1
            operator.append(row)
        operator = torch.cat(operator).to(energy_dist.loc)
        mean = operator @ energy_dist.loc
        covar = operator @ energy_dist.covariance_matrix @ operator.transpose(1, 0)
        return mean, covar
    
    def compute_forces_uncertainty_molecule(self, covar, inputs):
        assert max(inputs.batch) == 0, "Only batchsize of one works"
        n_atoms = len(inputs.pos)

        ones = torch.ones((n_atoms*3, 1))
        term_1 = torch.ones((n_atoms*3,n_atoms*3)) * covar[0, 0].cpu().detach().numpy()
        term_3 = (ones @ covar[0:1, 1:].cpu().detach().numpy())
        term_2 = (covar[1:, 0:1].cpu().detach().numpy() @ ones.transpose(0, 1).numpy())
        term_4 = covar[1:, 1:].cpu().detach().numpy()
        # analytic_covar = 1 / (self.epsilon**2) * (term_1 - term_2 - term_3 + term_4)
        analytic_covar = (term_1 - term_2 - term_3 + term_4)
        det = torch.det(analytic_covar)
        trace = torch.trace(analytic_covar) / len(analytic_covar)
        #largest_eig = torch.linalg.eig(analytic_covar)[0][0]
        det = torch.max(analytic_covar)
        min_val = torch.min(analytic_covar)
        largest_eig = torch.mean(analytic_covar)
        
        debug = False
        if debug:
            def get_gradient(f_sample):
                grad = []
                for i in range(1, len(f_sample)):
                    grad.append((f_sample[i] - f_sample[0]) )#/ self.epsilon)
                return torch.tensor(grad)
            f_samples = 0#energy_dist.sample(torch.Size([10000]))
            grads = []
            for f_sample in f_samples:
                grads.append(get_gradient(f_sample))
            grads = torch.cat(grads).reshape(10000, n_atoms*3)
            emp_cov = np.cov(grads.numpy().transpose())
        
        return det, trace, largest_eig#, max_val, min_val, mean_val