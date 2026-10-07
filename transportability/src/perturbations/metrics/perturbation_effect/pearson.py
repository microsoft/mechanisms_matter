"""Pearson-based perturbation-effect metrics."""

# pyright: reportUnknownMemberType=false
from typing import Any, cast

import numpy as np
from scipy.stats import pearsonr  # type: ignore


def pearson_pert(
    X_obs: np.ndarray,
    X_pred: np.ndarray,
    reference: np.ndarray,
    DEGs: np.ndarray | None = None,
    log_fold: bool = False,
    eps: float = 1e-6,
) -> float:
    """
    Compute Pearson using a specific reference, for one perturbation.

    Args:
        X_obs: Observed post-perturbation profile with shape ``(n_genes,)``.
        X_pred: Predicted post-perturbation profile with shape ``(n_genes,)``.
        reference: Reference profile with shape ``(n_genes,)``.
        DEGs: Optional boolean DEG mask with shape ``(n_genes,)``.
        log_fold: Whether to compare log-fold changes relative to ``reference``.
        eps: Small constant used to avoid taking ``log2(0)``.

    Returns:
        Pearson correlation between observed and predicted perturbation deltas.
    """
    if log_fold:
        delta_obs = np.log2(X_obs + eps) - np.log2(reference + eps)
        delta_pred = np.log2(X_pred + eps) - np.log2(reference + eps)
    else:
        delta_obs = X_obs - reference
        delta_pred = X_pred - reference

    if DEGs is not None:
        if DEGs.sum() >= 2:
            delta_obs = delta_obs[DEGs]
            delta_pred = delta_pred[DEGs]
        else:
            # if there is only 1 DEG, we cannot compute a meaningful Pearson correlation
            return np.nan

    if np.std(delta_obs) > 1e-6 and np.std(delta_pred) > 1e-6:
        return float(cast(Any, pearsonr(delta_obs, delta_pred))[0])
        # return np.corrcoef(delta_obs, delta_pred)[0, 1]
    else:
        return np.nan
