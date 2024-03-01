import numpy as np
from numpy.typing import NDArray
import torch
from scipy.stats import norm
from src.models.interfaces import BaseForceModel
from torch_geometric.data import Data, Batch

from .distances import average_l2


class CenterSmoothedForceModel(BaseForceModel):
    """Wrapper for BaseForceModels that enables center smoothed prediction and certification.

    See "Center Smoothing: Certified Robustness for Networks with Structured Outputs"
    (Kumar et al. 2022).
    """

    supported_d_out = {
        'average_l2': average_l2
    }

    def __init__(self, base_model: BaseForceModel,
                 std: float, alpha_1: float, alpha_2: float, delta: float,
                 n_samples_pred: int, n_samples_cert: int,
                 sample_batch_size: int, d_out: str = 'average_l2') -> None:
        """
        Args:
            base_model (BaseForceModel): The base model to wrap.
            std (float): Gaussian smoothing standard deviation.
            alpha_1 (float): Significance of prediction confidence interval.
            alpha_2 (float): Significance of certification confidence interval.
            delta (float): Approximation constant for minimum enclosing ball.
            n_samples_pred (int): Number of samples to use for prediction and abstention.
            n_samples_cert (int): Number of samples to use for certification.
            sample_batch_size (int): Number of samples to evaluate simultaneously.
            d_out (str): Smoothing distance to use.
                Should be in  ['average_l2']. Defaults to 'average_l2'.

        Raises:
            ValueError: If d_out not in ['average_l2'].
        """
        assert isinstance(base_model, BaseForceModel)
        super().__init__(base_model.model, base_model.loss_fn)

        self.std = std
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.delta = delta
        self.n_samples_pred = n_samples_pred
        self.n_samples_cert = n_samples_cert
        self.sample_batch_size = sample_batch_size

        if d_out not in self.supported_d_out:
            raise ValueError(f'Only support output distances {self.supported_d_out}')
        else:
            self.d_out = self.supported_d_out[d_out]

    def forward(self, batch: Batch) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Adds single Gaussian Gaussian sample to coords of all batch elements, applies base model.

        Args:
            batch (Batch): PyTorch Geometric Batch, where each element has attribute 'pos',
                where 'pos' is a Tensor of arbitrary shape (but usally N x 3).

        Returns:
            tuple[torch.FloatTensor, torch.FloatTensor]: Pair of predicted energies and forces.
                Shape should be [B, ] and [B x N x 3], respectively.
        """

        if self.std > 0:
            noisy_batch = batch.clone()
            noisy_batch.apply(lambda x: torch.normal(x, self.std), 'pos')
        else:
            noisy_batch = batch

        return super().forward(noisy_batch)

    def sample_force_batch(self, input_batch: Batch, n_samples: int) -> torch.FloatTensor:
        """Predicts forces for multiple noisy samples added to a batches' molecule coordinates.

        Currently only supports batch size 1.

        Args:
            input_batch (Batch): Batch of length 1, where each element has
                coordinates attribute pos ([N x 3])
                and atom charge attribute z ([N,]).
            n_samples (int): Number of samples to take

        Raises:
            NotImplementedError: When batch size is larger than 1.

        Returns:
            torch.FloatTensor: Predicted force vectors per sample and atom. 
                Shape should be [n_samples, N, 3].
        """
        if len(input_batch) > 1:
            raise NotImplementedError('Currently only support sampling for single molecule')

        molecule = input_batch[0]
        n_atoms = len(molecule.pos)
        repeated_pos = molecule.pos.tile(n_samples, 1).detach()
        repeated_z = molecule.z.tile(n_samples).detach()
        batch_idx = torch.arange(n_samples).repeat_interleave(n_atoms).detach()
        batch_ptr = (torch.arange(n_samples) * n_atoms).detach()

        if next(self.parameters()).is_cuda:
            repeated_pos = repeated_pos.cuda()
            repeated_z = repeated_z.cuda()
            batch_idx = batch_idx.cuda()
            batch_ptr = batch_ptr.cuda()

        repeated_batch = Data(pos=repeated_pos, z=repeated_z, batch=batch_idx, ptr=batch_ptr)

        _, forces = self.forward(repeated_batch)

        return forces.view(n_samples, n_atoms, -1)

    def pred_center(self, input_batch: Batch) -> tuple[torch.FloatTensor, float]:
        """Center smoothed prediction.

        Lines 1-5 of Algorithm 1 from (Kumar et al., 2022).
        Currently only supports batch size 1.

        Args:
            input_batch (Batch): Batch of length 1, where each element has
                coordinates attribute pos ([N x 3])
                and atom charge attribute z ([N,]).

        Raises:
            NotImplementedError: When batch size is larger than 1.

        Returns:
            tuple[torch.FloatTensor, float]: Predicted center and radius of 2-MEB.
                Center should have shape [N, 3].
                Radius of 2-Minimum-Enclosing-Ball is median output distance to center.
        """
        if len(input_batch) > 1:
            raise NotImplementedError('Currently only support sampling for single molecule')

        n_samples_remaining = self.n_samples_pred

        while n_samples_remaining > 0:
            n_batch_samples = min(n_samples_remaining, self.sample_batch_size)
            batch_pred = self.sample_force_batch(input_batch, n_batch_samples).detach().cpu()

            if n_samples_remaining == self.n_samples_pred:
                sampled_preds = batch_pred
            else:
                sampled_preds = torch.cat([sampled_preds, batch_pred], dim=0)

            n_samples_remaining -= n_batch_samples

        distances = torch.zeros((self.n_samples_pred, self.n_samples_pred))

        for i in range(1, self.n_samples_pred):
            distances[i, :i] = self.d_out(sampled_preds[i], sampled_preds[:i])

        # Make lower triangular matrix into a symmetric matrix
        distances = distances + distances.T
        median_distances = torch.median(distances, dim=1)[0]
        center_index = torch.argmin(median_distances, dim=0)

        return sampled_preds[center_index].detach(), median_distances[center_index].detach()

    def abstain(self, input_batch: Batch, center: torch.FloatTensor, radius: float) -> bool:
        """Determines whether to abstain from prediction.

        Lines 6-9 of Algorithm 1 in (Kumar et al., 2019).
        Currently only supports batch size 1.

        Args:
            input_batch (Batch): Batch of length 1, where each element has
                coordinates attribute pos ([N x 3])
                and atom charge attribute z ([N,]).
            center (torch.FloatTensor): Smoothed center prediction from pred_center().
            radius (float): Radius of 2-Minimum-Enclosing-Ball.

        Raises:
            NotImplementedError: When batch size is larger than 1.

        Returns:
            bool: Whether original center prediction is consistent with smoothed classifier.
                True < --> Abstain.
        """
        if len(input_batch) > 1:
            raise NotImplementedError('Currently only support sampling for single molecule')

        if next(self.parameters()).is_cuda:
            center = center.clone().cuda()
            radius = radius.clone().cuda()

        n_samples_remaining = self.n_samples_pred

        within_ball = 0

        while n_samples_remaining > 0:
            n_batch_samples = min(n_samples_remaining, self.sample_batch_size)
            batch_pred = self.sample_force_batch(input_batch, n_batch_samples).detach()

            dists_to_center = self.d_out(center, batch_pred)
            within_ball += (dists_to_center <= radius).sum().cpu()

            n_samples_remaining -= n_batch_samples

        frac_within_ball = within_ball / self.n_samples_pred

        delta_1 = np.sqrt(np.log(2 / self.alpha_1) / (2 * self.n_samples_pred))
        p_delta_1 = frac_within_ball - delta_1
        delta_2 = (1 / 2) - p_delta_1

        # True if should abstain
        return max(delta_1, delta_2) > self.delta

    def certify(self, input_batch: Batch, center: torch.FloatTensor,
                budgets: list[float]) -> torch.FloatTensor:
        """Computes certified output radius for list of adversarial perturbation budgets.

        Algorithm 2 in (Kumar et al., 2019).
        Currently only supports batch size 1.

        Args:
            input_batch (Batch): Batch of length 1, where each element has
                coordinates attribute pos ([N x 3])
                and atom charge attribute z ([N,]).
            center (torch.FloatTensor): Smoothed center prediction from pred_center().
            budgets (list[float]): List of adversarial budgets, i.e., l_2 perturbation radii.

        Raises:
            NotImplementedError: When batch size is larger than 1.

        Returns:
            torch.FloatTensor: Tensor of same length as budgets,
                with each entry specifying certified output distance for given input budget.
        """

        if len(input_batch) > 1:
            raise NotImplementedError('Currently only support sampling for single molecule')

        if next(self.parameters()).is_cuda:
            center = center.clone().cuda()

        n_samples_remaining = self.n_samples_cert

        while n_samples_remaining > 0:
            n_batch_samples = min(n_samples_remaining, self.sample_batch_size)
            batch_pred = self.sample_force_batch(input_batch, n_batch_samples).detach()

            batch_dists = self.d_out(center, batch_pred).detach().cpu()

            if n_samples_remaining == self.n_samples_cert:
                dists = batch_dists
            else:
                dists = torch.cat([dists, batch_dists])

            n_samples_remaining -= n_batch_samples

        dists = torch.sort(dists)[0]  # sort sampled distances in ascending order
        empirical_cdf = np.linspace(0, 1, self.n_samples_cert + 1)[1:]

        certified_output_distances = []
        for budget in budgets:
            quantile = self.calc_quantile(budget)
            assert quantile >= 0

            if budget == 0:
                output_distance = 0
            elif np.isclose(budget, self.calc_quantile(self.calc_max_rad())):
                output_distance = dists[-1]
            elif quantile > 1:
                output_distance = torch.inf
            else:
                output_distance = dists[empirical_cdf >= quantile].min()
                output_distance *= 3  # (1+beta) from paper

            certified_output_distances.append(output_distance)

        return torch.Tensor(certified_output_distances).detach()

    def calc_max_rad(self):
        """Maximum certifiable perturbation budget, given number of samples"""
        confidence_correction = np.sqrt(np.log(1 / self.alpha_2)/(2 * self.n_samples_cert))
        return self.std * (norm.ppf(1-confidence_correction) - norm.ppf(0.5 + self.delta))

    def calc_quantile(self, budget: float):
        """Computes upper bound on output distribution quantile function, given perturbation budget.

        Corresponds to lines 6-7 in Algorithm 2 of (Kumar et al., 2019).

        Args:
            budget (float): l_2 perturbation budget for input molecule.

        Returns:
            _type_: Upper bound on quantile function.
        """
        confidence_correction = np.sqrt(np.log(1 / self.alpha_2)/(2 * self.n_samples_cert))
        return norm.cdf(norm.ppf(0.5 + self.delta) + (budget / self.std)) + confidence_correction
