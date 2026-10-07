"""Implements various loss functions for training state-gene perturbation models, including Wasserstein distance, KL divergence, MMD, and a combined tabular loss. These losses are designed to measure the distance between predicted and target distributions of gene expression profiles, accounting for both gene-level and cell-level differences."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class WassersteinLoss(nn.Module):
    """Implements Wasserstein distance loss for distributions represented by logits.This implementation supports both 1D and 2D Wasserstein distance calculations."""

    def __init__(self, p: int = 1, reduction: str = "mean"):
        """
        Constructor for WassersteinLoss.

        Args:
            p (int): Order of Wasserstein distance (1 or 2)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()  # type: ignore
        self.p = p
        self.reduction = reduction

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute Wasserstein distance between predicted and target distributions.

        Args:
            p (torch.Tensor): Predicted logits of shape (batch_size, num_classes)
            q (torch.Tensor): Target probabilities of shape (batch_size, num_classes)
                                 or class indices of shape (batch_size,)

        Returns:
            torch.Tensor: Computed Wasserstein distance
        """
        q = torch.nan_to_num(q, nan=0.0)
        # Convert logits to probabilities
        pred_probs = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Compute cumulative distribution functions (CDFs)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(q, dim=-1)

        max_len = max(pred_cdf.size(1), target_cdf.size(1))
        if pred_cdf.size(1) < max_len:
            pred_cdf = F.pad(pred_cdf, (0, max_len - pred_cdf.size(1)), "constant", 0)
        if target_cdf.size(1) < max_len:
            target_cdf = F.pad(target_cdf, (0, max_len - target_cdf.size(1)), "constant", 0)

        # Compute Wasserstein distance
        wasserstein_dist = torch.abs(pred_cdf - target_cdf).pow(self.p)
        wasserstein_dist = wasserstein_dist.sum(dim=-1)

        # Apply reduction if specified
        if self.reduction == "mean":
            return wasserstein_dist.mean()
        elif self.reduction == "sum":
            return wasserstein_dist.sum()
        return wasserstein_dist


class KLDivergenceLoss(nn.Module):
    """Implements KL divergence loss for distributions represented by logits. This implementation computes the KL divergence between two distributions, with optional normalization to convert logits to probabilities."""

    def __init__(self, apply_normalization: bool = False, epsilon: float = 1e-10):
        """Constructor for KLDivergenceLoss."""
        super().__init__()  # type: ignore
        self.apply_normalization = apply_normalization
        self.epsilon = epsilon

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between predicted and target distributions."""
        q = torch.nan_to_num(q, nan=0.0)
        p = torch.nan_to_num(p, nan=0.0)

        max_len = max(p.size(1), q.size(1))
        if p.size(1) < max_len:
            p = F.pad(p, (0, max_len - p.size(1)), "constant", 0)
        if q.size(1) < max_len:
            q = F.pad(q, (0, max_len - q.size(1)), "constant", 0)

        if self.apply_normalization:
            p = F.softmax(p, dim=-1)
            q = F.softmax(q, dim=-1)

        return torch.sum(p * torch.log(p / q))


class MMDLoss(nn.Module):
    """Implements Maximum Mean Discrepancy (MMD) loss using the geomloss library. This loss measures the distance between two distributions based on their samples, and can be used to compare predicted and target gene expression profiles."""

    def __init__(
        self, kernel: str = "energy", blur: float = 0.05, scaling: float = 0.5, downsample: int = 1
    ):
        """Constructor for MMDLoss."""
        super().__init__()  # type: ignore
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)
        self.downsample = downsample

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MMD loss between input and target distributions."""
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])

        loss = self.mmd_loss(input, target)
        return loss.mean()


class TabularLoss(nn.Module):
    """Implements a combined loss function for tabular data, specifically designed for state-gene perturbation models. This loss combines gene-level and cell-level MMD losses to capture both gene expression differences and cell state differences between predicted and target distributions."""

    def __init__(self, shared: int = 128, downsample: int = 1):
        """Constructor for TabularLoss."""
        super().__init__()  # type: ignore
        self.shared = shared
        self.downsample = downsample

        self.gene_loss = SamplesLoss(loss="energy")
        self.cell_loss = SamplesLoss(loss="energy")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined gene-level and cell-level MMD loss."""
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])
        gene_mmd = self.gene_loss(input, target).nanmean()

        # cell_mmd should only be on the shared genes, and match scale to mse loss
        cell_inputs = input[:, :, -self.shared :]
        cell_targets = target[:, :, -self.shared :]

        # need to reshape each from (B, self.downsample, F) to (F, self.downsample, B)
        cell_inputs = cell_inputs.transpose(2, 0)
        cell_targets = cell_targets.transpose(2, 0)
        cell_mmd = self.cell_loss(cell_inputs, cell_targets).nanmean()

        final_loss = torch.tensor(0.0).to(cell_mmd.device)
        if not gene_mmd.isnan():
            final_loss += gene_mmd
        if not cell_mmd.isnan():
            final_loss += cell_mmd

        return final_loss
