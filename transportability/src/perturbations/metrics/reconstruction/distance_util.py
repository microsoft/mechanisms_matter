"""Shared kernel and distance helpers for reconstruction metrics."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.experimental import AnnCollection
from scipy.linalg import sqrtm  # type: ignore[import]
from scipy.stats import nbinom, poisson
from sklearn.metrics.pairwise import pairwise_kernels  # type: ignore[import]

from perturbations.util.anndata_util import (
    extract_rows,
    fit_control_incremental_pca,
    obs_has_key,
)

_EPS = 1e-12
DEFAULT_GAMMA_ESTIMATION_MAX_SAMPLES = 512


def _as_2d_array(x: np.ndarray, name: str) -> np.ndarray:
    """Return a float64 2D view of an array-like input."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape={arr.shape}.")
    return arr


def _validate_feature_dims(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired matrices have matching feature dimensions."""
    x = _as_2d_array(x, "x")
    y = _as_2d_array(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "x and y must have the same number of dimensions (features). "
            + f"Got {x.shape[1]} and {y.shape[1]}."
        )
    return x, y


def pairwise_squared_distances(
    x: np.ndarray,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Pairwise squared Euclidean distances computed from dot products."""
    x = _as_2d_array(x, "x")
    if y is None:
        y = x
    else:
        y = _as_2d_array(y, "y")
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                "x and y must have the same number of dimensions (features). "
                + f"Got {x.shape[1]} and {y.shape[1]}."
            )

    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    distances = cast(np.ndarray, x_norm + y_norm - 2.0 * (x @ y.T))
    np.maximum(distances, 0.0, out=distances)
    return distances


def median_positive(values: np.ndarray, default: float = 1.0) -> float:
    """Median of strictly positive values, or a default when unavailable."""
    vals = np.asarray(values, dtype=np.float64)
    positive = vals[vals > 0.0]
    if positive.size == 0:
        return float(default)
    return float(np.median(positive))


def kernel_mean(
    x: np.ndarray,
    y: np.ndarray | None = None,
    kernel: str = "rbf",
    **kernel_params: Any,
) -> float:
    """Mean kernel value over all row pairs."""
    x = _as_2d_array(x, "x")
    if y is None:
        y = x
    else:
        x, y = _validate_feature_dims(x, y)

    kernel_matrix = pairwise_kernels(
        x,
        y,
        metric=kernel,
        filter_params=True,
        **kernel_params,
    )
    return float(kernel_matrix.mean())


def biased_mmd2_from_kernel_means(
    xx_mean: float,
    yy_mean: float,
    xy_mean: float,
) -> float:
    """Biased MMD^2 from mean within/between-group kernel values."""
    return float(max(xx_mean + yy_mean - 2.0 * xy_mean, 0.0))


def mmd2(
    x_obs: np.ndarray,
    x_pred: np.ndarray,
    kernel: str = "rbf",
    estimator: str = "unbiased",
    **kernel_params: Any,
) -> float:
    """Maximum Mean Discrepancy squared for two sample sets."""
    x_obs, x_pred = _validate_feature_dims(x_obs, x_pred)
    m, n = x_obs.shape[0], x_pred.shape[0]

    Kxx = pairwise_kernels(
        x_obs,
        x_obs,
        metric=kernel,
        filter_params=True,
        **kernel_params,
    )
    Kyy = pairwise_kernels(
        x_pred,
        x_pred,
        metric=kernel,
        filter_params=True,
        **kernel_params,
    )
    Kxy = pairwise_kernels(
        x_obs,
        x_pred,
        metric=kernel,
        filter_params=True,
        **kernel_params,
    )

    if estimator == "biased":
        return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    if estimator != "unbiased":
        raise ValueError(f"Unsupported estimator: {estimator}. Expected 'unbiased' or 'biased'.")

    if m >= 2 and n >= 2:
        xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
        yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        xy = Kxy.mean()
        return float(xx + yy - 2.0 * xy)

    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def mmd_distance(
    x_obs: np.ndarray,
    x_pred: np.ndarray,
    kernel: str = "rbf",
    **kernel_params: Any,
) -> float:
    """MMD distance computed from the unbiased MMD^2 estimator."""
    return float(
        np.sqrt(
            max(
                mmd2(x_obs, x_pred, kernel=kernel, estimator="unbiased", **kernel_params),
                0.0,
            )
        )
    )


def sample_covariance(x: np.ndarray) -> np.ndarray:
    """Estimate a sample covariance matrix with finite output for tiny batches."""
    x = _as_2d_array(x, "x")
    n_samples, n_features = x.shape
    if n_samples <= 1:
        return np.zeros((n_features, n_features), dtype=np.float64)

    covariance = np.cov(x, rowvar=False)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return covariance


def fid_distance(
    x_obs: np.ndarray,
    x_pred: np.ndarray,
    covariance_eps: float = 1e-6,
) -> float:
    """Fréchet distance between observed and predicted samples."""
    x_obs, x_pred = _validate_feature_dims(x_obs, x_pred)

    mu_obs = np.mean(x_obs, axis=0, dtype=np.float64)
    mu_pred = np.mean(x_pred, axis=0, dtype=np.float64)
    sigma_obs = sample_covariance(x_obs)
    sigma_pred = sample_covariance(x_pred)

    eye = np.eye(x_obs.shape[1], dtype=np.float64)
    sigma_obs_reg = sigma_obs + covariance_eps * eye
    sigma_pred_reg = sigma_pred + covariance_eps * eye

    covmean_arr = np.asarray(cast(Any, sqrtm(sigma_obs_reg @ sigma_pred_reg)))
    covmean_arr = np.real_if_close(covmean_arr, tol=1000)
    if np.iscomplexobj(covmean_arr):
        covmean_arr = np.real(covmean_arr)
    covmean = np.asarray(covmean_arr, dtype=np.float64)

    mean_diff = np.sum((mu_obs - mu_pred) ** 2, dtype=np.float64)
    cov_diff = np.trace(sigma_obs_reg + sigma_pred_reg - 2.0 * covmean)
    return max(float(mean_diff + cov_diff), 0.0)


def subsampling(
    matrix: np.ndarray,
    max_rows: int,
    seed: int | None = 0,
) -> np.ndarray:
    """Randomly subsample rows without replacement."""
    n_rows = int(matrix.shape[0])
    if n_rows <= max_rows:
        return matrix
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    return matrix[idx]


def adaptive_gamma(x_obs: np.ndarray) -> float:
    """Estimate RBF gamma from positive pairwise squared distances."""
    x_obs = _as_2d_array(x_obs, "x_obs")
    if x_obs.shape[0] < 2:
        return 1.0 / max(int(x_obs.shape[1]), 1)

    dist_sq = pairwise_squared_distances(x_obs)
    tri = np.triu_indices(int(x_obs.shape[0]), k=1)
    median_sqdist = median_positive(dist_sq[tri], default=0.0)
    if median_sqdist <= 0.0:
        return 1.0 / max(int(x_obs.shape[1]), 1)
    return 1.0 / max(median_sqdist, _EPS)


def estimate_mmd_gamma(
    obs: AnnData | AnnCollection,
    layer_obs: str | None,
    control_label: str | int | float = -1,
    use_pca: bool = False,
    n_pca_components: int = 50,
    ipca_batch_size: int = 1024,
    seed: int | None = 0,
    gamma_estimation_max_samples: int = DEFAULT_GAMMA_ESTIMATION_MAX_SAMPLES,
    pca_model: Any | None = None,
) -> float:
    """
    Estimate a shared RBF gamma from observed non-control cells.

    This uses the observed-only preprocessing policy shared by reconstruction
    metrics: optional control-fitted PCA followed by the median positive
    pairwise squared distance heuristic on non-control observed cells.
    """
    if not obs_has_key(obs.obs, "perturbation"):
        raise KeyError("obs.obs must contain a 'perturbation' column.")

    if pca_model is None and use_pca:
        pca_model = fit_control_incremental_pca(
            data_obj=obs,
            layer_key=layer_obs,
            control_label=control_label,
            n_pca_components=n_pca_components,
            batch_size=ipca_batch_size,
            obs_key="perturbation",
            data_name="obs",
        )

    obs_df = cast(pd.DataFrame, obs.obs)
    obs_labels = np.asarray(obs_df["perturbation"])
    obs_reference_idx = np.flatnonzero(obs_labels != control_label)
    sampled_obs_idx = subsampling(
        obs_reference_idx,
        max_rows=max(int(gamma_estimation_max_samples), 2),
        seed=seed,
    )
    x_reference = extract_rows(obs, sampled_obs_idx, layer_obs)
    if pca_model is not None:
        x_reference = pca_model.transform(x_reference).astype(np.float64, copy=False)

    return adaptive_gamma(x_reference)


def param_distance(
    x_obs: np.ndarray,
    x_pred: np.ndarray,
    parametric_form: str = "NB",
    dist_type: str = "JS-divergence",
) -> float:
    """
        Calculate parametric distribution distance between observed and predicted values.

        For each gene, this function fits the selected parametric family to observed
        and predicted samples, computes a per-gene distance, and returns the mean
        distance across genes.

    Arguments:
        x_obs: observed post-perturbation profile. Shape: (n_cells, n_genes)
        x_pred: predicted post-perturbation profile. Shape: (n_cells, n_genes)
        parametric_form: "NB" (Negative Binomial), "Poisson", "ZINB" (Zero-Inflated NB).
            NB/ZINB are fit in the common (mu, theta) parameterization:
                Var = mu + mu^2 / theta
        dist_type: "JS-divergence" or "Wasserstein" (1-Wasserstein on counts).

    Returns:
        Mean per-gene distribution distance as a float.
    """
    q_tail = 1.0 - 1e-8  # truncation quantile
    kmax_cap = 10_000  # safety cap to avoid huge loops

    def _fit_poisson(mu: float):
        lam = max(float(mu), 0.0)
        return {"lam": lam}

    def _fit_nb(mu: float, var: float):
        mu = max(float(mu), 0.0)
        var = max(float(var), 0.0)
        # Method-of-moments for theta: var = mu + mu^2/theta  => theta = mu^2/(var-mu)
        denom = max(var - mu, 0.0)
        if mu <= _EPS:
            theta = 1e8
        elif denom <= 1e-8 * max(mu, 1.0):  # near-Poisson
            theta = 1e8
        else:
            theta = (mu * mu) / max(denom, _EPS)
        theta = float(np.clip(theta, 1e-8, 1e12))
        return {"mu": mu, "theta": theta}

    def _nbinom_from_mu_theta(mu: float, theta: float):
        # SciPy nbinom(n, p): mean = n*(1-p)/p
        n = float(theta)
        p = n / (n + float(mu) + _EPS)
        p = float(np.clip(p, _EPS, 1.0 - _EPS))
        n = float(max(n, _EPS))
        return n, p

    def _fit_zinb(mu: float, var: float, p0: float):
        # crude but stable: fit NB by moments, then set pi from "excess zeros"
        nb = _fit_nb(mu, var)
        n, p = _nbinom_from_mu_theta(nb["mu"], nb["theta"])
        nb_p0 = float(cast(Any, nbinom(n, p)).pmf(0))  # = p**n
        p0 = float(np.clip(p0, 0.0, 1.0))
        pi = (p0 - nb_p0) / max(1.0 - nb_p0, _EPS)
        pi = float(np.clip(pi, 0.0, 1.0 - 1e-12))
        return {"mu": nb["mu"], "theta": nb["theta"], "pi": pi}

    def _pmf(params: dict[str, float], ks: np.ndarray, form: str) -> np.ndarray:
        if form == "POISSON":
            pmf = np.asarray(cast(Any, poisson(params["lam"])).pmf(ks), dtype=float)
        elif form == "NB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pmf = np.asarray(cast(Any, nbinom(n, p)).pmf(ks), dtype=float)
        elif form == "ZINB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pi = params["pi"]
            pmf = (1.0 - pi) * np.asarray(cast(Any, nbinom(n, p)).pmf(ks), dtype=float)
            pmf[0] += pi
        else:
            raise ValueError(f"Unknown parametric_form: {form}")
        s = pmf.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback to a degenerate-at-zero distribution
            pmf = np.zeros_like(pmf)
            pmf[0] = 1.0
            return pmf
        return pmf / s

    def _ppf(params: dict[str, float], q: float, form: str) -> float:
        q = float(np.clip(q, _EPS, 1.0 - _EPS))
        if form == "POISSON":
            return float(cast(Any, poisson(params["lam"])).ppf(q))
        elif form == "NB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            return float(cast(Any, nbinom(n, p)).ppf(q))
        elif form == "ZINB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pi = params["pi"]
            nb_dist = nbinom(n, p)

            p0 = pi + (1.0 - pi) * float(cast(Any, nb_dist).cdf(0))
            if q <= p0:
                return 0.0

            q_nb = (q - pi) / max(1.0 - pi, _EPS)
            q_nb = float(np.clip(q_nb, _EPS, 1.0 - _EPS))
            return float(cast(Any, nb_dist).ppf(q_nb))
        else:
            raise ValueError(f"Unknown parametric_form: {form}")

    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        # KL(p||m) and KL(q||m)
        kl_pm = np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS)))
        kl_qm = np.sum(q * (np.log(q + _EPS) - np.log(m + _EPS)))
        return float(0.5 * (kl_pm + kl_qm))

    def _wasserstein_1(p: np.ndarray, q: np.ndarray) -> float:
        # Exact 1-Wasserstein on integers with unit spacing: sum_k |CDF_p(k)-CDF_q(k)|
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)))

    x_obs = np.asarray(x_obs, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(1, -1)
    if x_pred.ndim == 1:
        x_pred = x_pred.reshape(1, -1)

    if x_obs.ndim != 2 or x_pred.ndim != 2:
        raise ValueError(
            f"x_obs and x_pred must be 1D or 2D arrays, got {x_obs.ndim}D and {x_pred.ndim}D."
        )

    if x_obs.shape[1] != x_pred.shape[1]:
        raise ValueError(
            "x_obs and x_pred must have the same number of genes, "
            + f"got {x_obs.shape[1]} vs {x_pred.shape[1]}."
        )
    if x_obs.shape[0] == 0 or x_pred.shape[0] == 0:
        return float("nan")

    # Count models: clip negatives
    x_obs = np.clip(x_obs, 0.0, None)
    x_pred = np.clip(x_pred, 0.0, None)

    form = parametric_form.strip().upper()
    dist = dist_type.strip().upper()

    n_cells_obs, n_genes = x_obs.shape
    n_cells_pred = x_pred.shape[0]
    mu_obs = x_obs.mean(axis=0)
    mu_pred = x_pred.mean(axis=0)
    var_obs = x_obs.var(axis=0, ddof=1) if n_cells_obs > 1 else np.zeros(n_genes)
    var_pred = x_pred.var(axis=0, ddof=1) if n_cells_pred > 1 else np.zeros(n_genes)
    p0_obs = (x_obs == 0).mean(axis=0)
    p0_pred = (x_pred == 0).mean(axis=0)

    dists: list[float] = []

    for g in range(n_genes):
        if form == "POISSON":
            p_params = _fit_poisson(mu_obs[g])
            q_params = _fit_poisson(mu_pred[g])
        elif form == "NB":
            p_params = _fit_nb(mu_obs[g], var_obs[g])
            q_params = _fit_nb(mu_pred[g], var_pred[g])
        elif form == "ZINB":
            p_params = _fit_zinb(mu_obs[g], var_obs[g], p0_obs[g])
            q_params = _fit_zinb(mu_pred[g], var_pred[g], p0_pred[g])
        else:
            raise ValueError('parametric_form must be one of {"NB","Poisson","ZINB"}')

        # choose truncation support
        data_max = int(max(np.max(x_obs[:, g]), np.max(x_pred[:, g])))
        k1 = _ppf(p_params, q_tail, form)
        k2 = _ppf(q_params, q_tail, form)
        kmax = int(min(max(data_max, k1, k2), kmax_cap))
        ks = np.arange(kmax + 1, dtype=int)

        p_pmf = _pmf(p_params, ks, form)
        q_pmf = _pmf(q_params, ks, form)

        if dist in {"JS-DIVERGENCE", "JS"}:
            d = _js_divergence(p_pmf, q_pmf)
        elif dist in {"WASSERSTEIN", "WASSERSTEIN-1", "W1"}:
            d = _wasserstein_1(p_pmf, q_pmf)
        else:
            raise ValueError('dist_type must be "JS-divergence" or "Wasserstein"')

        if np.isfinite(d):
            dists.append(d)

    return float(np.mean(dists)) if len(dists) > 0 else float("nan")
