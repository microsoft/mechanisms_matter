"""Perturbation discrimination score metric implementations."""

import numpy as np
from sklearn.metrics import pairwise_distances


def pds(
    X_obs: np.ndarray,
    X_pred: np.ndarray,
    reference: np.ndarray,
    metric: str = "l1",
    log_fold: bool = False,
    eps: float = 1e-6,
) -> float:
    """
    Compute discrimination score for each perturbation.

    metric:
      - "l1", "l2": sklearn pairwise distances
      - "correlation"/"pearson": 1 - Pearson correlation
      - "cosine": sklearn cosine distance
      - "sign": 1 - (proportion of sign matches), computed over genes with nonzero true effect
    """
    if log_fold:
        X_obs = np.log2(X_obs + eps)
        X_pred = np.log2(X_pred + eps)
        reference = np.log2(reference + eps)

    true_effects = X_obs - reference
    pred_effects = X_pred - reference

    n_perts = true_effects.shape[0]
    scores = np.empty(n_perts)

    for i in range(n_perts):
        R = true_effects[:,]  # (n_perts, n_genes_sel)
        p = pred_effects[i,]  # (n_genes_sel,)

        m = metric.lower()
        if m in {"correlation", "pearson"}:
            M = np.vstack([R, p])
            C = np.corrcoef(M)
            corr = C[-1, :-1]
            corr = np.where(np.isnan(corr), -1.0, corr)
            distances = 1.0 - corr

        elif m == "sign":
            # sign-based distance: 1 - proportion of sign matches over nonzero true genes
            R_sign = np.sign(R)  # (n_perts, n_genes_sel)
            p_sign = np.sign(p)[None, :]  # (1, n_genes_sel) broadcast
            mask_nz = R != 0  # only count where true != 0
            agree = (R_sign == p_sign) & mask_nz  # correct sign & valid
            correct_counts = agree.sum(axis=1)
            denom = mask_nz.sum(axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                prop_match = np.where(denom > 0, correct_counts / denom, 0.0)
            distances = 1.0 - prop_match  # lower is better

        else:
            distances = pairwise_distances(R, p.reshape(1, -1), metric=m).flatten()

        # PDS (PerturBench /Virtual Cell Challenge):
        #   r_q = (1 / (Q-1)) * sum_{q' != q} I( d(pred_q, true_q) <= d(pred_q, true_q') )
        # Random => 0.5, perfect => 1.0, worst => 0.0.
        d_self = distances[i]
        d_others = np.delete(distances, i)
        matches = np.asarray(d_self <= d_others, dtype=bool)
        scores[i] = float(matches.mean())

    return float(np.mean(scores))
