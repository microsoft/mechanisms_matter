"""Metrics for evaluating reconstruction of post-perturbation profiles."""

import numpy as np


def mean_error_pert(
    x_obs: np.ndarray, x_pred: np.ndarray, type: str = "absolute", weights: np.ndarray | None = None
) -> float:
    """
    Calculate mean error between observed and predicted values, for one perturbation.

    Args:
        x_obs: observed post-perturbation profile. Shape: (n_genes,)
        x_pred: predicted post-perturbation profile. Shape: (n_genes,)
        type: Type of error to calculate ("absolute", "squared", or "root-mean-squared")
        weights: Optional weights for genes. Shape: (n_genes,)

    Returns:
        Mean error as a float
    """
    if x_obs.shape != x_pred.shape:
        raise ValueError(
            f"x_obs and x_pred must have the same shape; got {x_obs.shape} vs {x_pred.shape}."
        )

    diff = x_pred - x_obs

    if weights is None:
        if type == "absolute":
            return float(np.mean(np.abs(diff)))
        elif type == "squared":
            return float(np.mean(diff**2))
        elif type == "root-mean-squared":
            return float(np.sqrt(np.mean(diff**2)))
        else:
            raise ValueError(
                f"Unknown type={type!r}. Expected one of: 'absolute', 'squared', 'root-mean-squared'."
            )
    else:
        if weights.shape != x_obs.shape:
            raise ValueError(f"weights must have shape {x_obs.shape}; got {weights.shape}.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")

        normalized_weights = weights / np.sum(weights)

        if type == "absolute":
            return float(np.sum(normalized_weights * np.abs(diff)))
        elif type == "squared":
            return float(np.sum(normalized_weights * (diff**2)))
        elif type == "root-mean-squared":
            return float(np.sqrt(np.sum(normalized_weights * (diff**2))))
        else:
            raise ValueError(
                f"Unknown type={type!r}. Expected one of: 'absolute', 'squared', 'root-mean-squared'."
            )
