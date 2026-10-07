"""Sets up evaluation metrics and kernel functions for computing MMD and Wasserstein distances between distributions."""

# pyright: reportUnknownMemberType=false

import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import ot
import torch
from torch import nn


class RBFKernel(nn.Module):
    """Radial Basis Function (RBF) kernel module."""

    def __init__(self, scale: float = 1.0):
        """Initialize the RBF kernel with the given scale parameter."""
        super().__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two sets of vectors."""
        x_norm = (x**2).sum(dim=1, keepdim=True)  # Bx x 1
        y_norm = (y**2).sum(dim=1, keepdim=True)  # By x 1
        squared_ell_2 = x_norm - 2 * x @ y.T + y_norm.T  # Bx x By

        return torch.exp(-self.scale * squared_ell_2)


class BrayCurtisKernel(nn.Module):
    """Bray-Curtis kernel module."""

    def __init__(
        self,
    ):
        """Initialize the Bray-Curtis kernel module."""
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Bray-Curtis kernel between two sets of vectors."""
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.abs(x - y).sum(dim=2)  # Bx x By
        denominator = torch.abs(x + y).sum(dim=2) + 1e-8

        return 1 - numerator / denominator


class TanimotoKernel(nn.Module):
    """Tanimoto kernel module."""

    def __init__(
        self,
    ):
        """Initialize the Tanimoto kernel module."""
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Tanimoto kernel between two sets of vectors."""
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = (x * y).sum(dim=2)  # Bx x By
        denominator = (x + y - x * y).sum(dim=2) + 1e-8

        return numerator / denominator


class RuzickaKernel(nn.Module):
    """Ruzicka kernel module."""

    def __init__(
        self,
    ):
        """Initialize the Ruzicka kernel module."""
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Ruzicka kernel between two sets of vectors."""
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.min(x, y).sum(dim=2)  # Bx x By
        denominator = torch.max(x, y).sum(dim=2) + 1e-8

        return numerator / denominator


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy (MMD) loss module."""

    def __init__(self, kernel: nn.Module):
        """Initialize the MMD loss with the given kernel."""
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the MMD loss between two sets of vectors."""
        k_xx = self.kernel(x, x)
        k_yy = self.kernel(y, y)
        k_xy = self.kernel(x, y)

        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Literal["emd", "sinkhorn"] = "emd",
    reg: float = 0.05,
    power: int = 2,
) -> float:
    """Compute the Wasserstein distance between two sets of vectors using the specified method."""
    assert power == 1 or power == 2

    ot_fn: Callable[..., Any]
    if method == "emd":
        ot_fn = cast("Callable[..., Any]", ot.emd2)
    elif method == "sinkhorn":
        ot_fn = cast("Callable[..., Any]", partial(ot.sinkhorn2, reg=reg))
    else:
        raise ValueError(f"Unknown method: {method}")

    a = cast(Any, ot.unif(x0.shape[0], type_as=x0))
    b = cast(Any, ot.unif(x1.shape[0], type_as=x1))
    cost = torch.cdist(x0, x1)
    if power == 2:
        cost = cost**2
    ret = ot_fn(a, b, cost, numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return float(ret)
