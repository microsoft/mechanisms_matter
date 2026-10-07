"""R-squared perturbation-effect metric utilities."""

import numpy as np
from sklearn.metrics import r2_score


def r2_score_pert(
    X_obs: np.ndarray,
    X_pred: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
    log_fold: bool = False,
    eps: float = 1e-6,
) -> float:
    """
    Calculate R2 score between observed and predicted deltas, for one perturbation.

    Arguments:
        X_obs: observed post-perturbation profile. Shape: (n_genes,)
        X_pred: predicted post-perturbation profile. Shape: (n_genes,)
        reference: reference. Shape: (n_genes,)
        weights: Optional weights for genes. Shape: (n_genes,)
        log_fold: whether to log fold change before computing R2
        eps: small constant to avoid log of zero when log_fold is True

    Returns:
        r2 as a float
    """
    if log_fold:
        delta_obs = np.log2(X_obs + eps) - np.log2(reference + eps)
        delta_pred = np.log2(X_pred + eps) - np.log2(reference + eps)
    else:
        delta_obs = X_obs - reference
        delta_pred = X_pred - reference

    if len(delta_obs) < 2 or len(delta_pred) < 2 or delta_obs.shape != delta_pred.shape:
        return np.nan
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != delta_obs.shape:
            return np.nan
        if np.count_nonzero(weights > 0) < 2:
            return np.nan
        return r2_score(delta_obs, delta_pred, sample_weight=weights)
    else:
        return r2_score(delta_obs, delta_pred)
