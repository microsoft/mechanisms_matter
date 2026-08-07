"""Distribution-distance metrics for perturbation reconstruction evaluation."""

# pyright: reportUnknownMemberType=false

from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.experimental import AnnCollection

from perturbations.metrics.reconstruction.distance_util import (
    fid_distance,
    mmd_distance,
    param_distance,
)
from perturbations.util.anndata_util import (
    extract_rows,
    fit_control_incremental_pca,
    obs_has_key,
)


@overload
def distribution_distance(
    obs: AnnData | AnnCollection,
    pred: AnnData | AnnCollection,
    layer_obs: str | None,
    layer_pred: str | None,
    control_label: str | int | float = ...,
    method: str = ...,
    distribution_form: str = ...,
    dist_type: str = ...,
    kernel: str = ...,
    use_pca: bool = ...,
    n_pca_components: int = ...,
    ipca_batch_size: int = ...,
    aggregate: str = ...,
    pca_model: Any | None = ...,
    return_details: Literal[False] = ...,
    **kernel_params: Any,
) -> float: ...


@overload
def distribution_distance(
    obs: AnnData | AnnCollection,
    pred: AnnData | AnnCollection,
    layer_obs: str | None,
    layer_pred: str | None,
    control_label: str | int | float = ...,
    method: str = ...,
    distribution_form: str = ...,
    dist_type: str = ...,
    kernel: str = ...,
    use_pca: bool = ...,
    n_pca_components: int = ...,
    ipca_batch_size: int = ...,
    aggregate: str = ...,
    pca_model: Any | None = ...,
    *,
    return_details: Literal[True],
    **kernel_params: Any,
) -> tuple[float, dict[str, float]]: ...


def distribution_distance(
    obs: AnnData | AnnCollection,
    pred: AnnData | AnnCollection,
    layer_obs: str | None,
    layer_pred: str | None,
    control_label: str | int | float = -1,
    method: str = "parametric",
    distribution_form: str = "NB",
    dist_type: str = "JS-divergence",
    kernel: str = "rbf",
    use_pca: bool = False,
    n_pca_components: int = 50,
    ipca_batch_size: int = 1024,
    aggregate: str = "median",
    pca_model: Any | None = None,
    return_details: bool = False,
    **kernel_params: Any,
) -> float | tuple[float, dict[str, float]]:
    """
    Calculate a distribution distance between observed and predicted profiles.

    Two methods are supported:
        1) ``parametric``: fit per-gene parametric distributions and compare them.
        2) ``mmd``: compute a kernel Maximum Mean Discrepancy between samples.
        3) ``fid``: compute a Fréchet distance between sample Gaussians.

    The function computes one distance per perturbation and aggregates those
    distances across perturbations.

    Args:
        obs: Observed post-perturbation profiles.
                obs: observed post-perturbation profile, an object of AnnData or AnnCollection.
        pred: Predicted post-perturbation profiles.
        pred: predicted post-perturbation profile, an object of AnnData or AnnCollection
        layer_obs: Layer key in ``obs`` used for distance calculation.
            Usually a normalized log-expression layer.
            Set to ``None`` to read from ``.X``.
        layer_pred: Layer key in ``pred`` used for distance calculation.
            Usually a normalized log-expression layer.
            Set to ``None`` to read from ``.X``.
        control_label: Label indicating control samples to exclude.
        method: Distance method, one of ``"parametric"``, ``"mmd"``, or ``"fid"``.
        distribution_form: Parametric family when ``method="parametric"``.
        dist_type: Distance type when ``method="parametric"``.
        kernel: Kernel name used when ``method="mmd"``.
        use_pca: Whether to fit/control-transform with IncrementalPCA first.
        n_pca_components: Number of PCA components when ``use_pca=True``.
        ipca_batch_size: Batch size for IncrementalPCA ``partial_fit``.
        aggregate: Aggregation across perturbations, either ``"median"`` or
            ``"mean"``.
        pca_model: Optional precomputed PCA model used to transform obs/pred
            before ``"mmd"`` or ``"fid"`` calculation.
        return_details: Whether to return a dictionary of per-perturbation distances
            in addition to the aggregated distance.
        **kernel_params: Extra kernel parameters passed to ``pairwise_kernels``.

    Returns:
        A float representing the distribution distance between obs and pred,
        or a tuple of the float and a dictionary of detailed distances if
        ``return_details=True``.
    """
    method = method.strip().lower()
    aggregate = aggregate.strip().lower()
    if use_pca and method not in {"mmd", "fid"}:
        use_pca = False
        print(
            f"Warning: use_pca is only implemented for method in {'mmd', 'fid'}. "
            + "Falling back to use_pca=False."
        )
    if pca_model is not None and method not in {"mmd", "fid"}:
        pca_model = None
        print(
            f"Warning: pca_model is only used for method in {'mmd', 'fid'}. "
            + "Ignoring the provided pca_model."
        )
    if aggregate not in {"median", "mean"}:
        raise ValueError(f"Unsupported aggregate: {aggregate}. Expected 'median' or 'mean'.")

    for name, data_obj in (("obs", obs), ("pred", pred)):
        obs_df = cast(pd.DataFrame, data_obj.obs)
        if not obs_has_key(obs_df, "perturbation"):
            raise KeyError(f"{name}.obs must contain a 'perturbation' column.")

    if obs.n_vars != pred.n_vars:
        raise ValueError(
            f"obs and pred must have the same number of genes. Got {obs.n_vars} vs {pred.n_vars}."
        )

    obs_labels = np.asarray(obs.obs["perturbation"])  # type: ignore[assignment]
    pred_labels = np.asarray(pred.obs["perturbation"])  # type: ignore[assignment]

    # remove control labels, based on the control_label value
    obs_perts = set(np.unique(obs_labels)) - {control_label}
    pred_perts = set(np.unique(pred_labels)) - {control_label}

    if obs_perts != pred_perts:
        missing_in_pred = obs_perts - pred_perts
        missing_in_obs = pred_perts - obs_perts
        raise ValueError(
            "obs and pred have mismatched perturbation labels (excluding control). "
            + f"Missing in pred: {missing_in_pred}; missing in obs: {missing_in_obs}."
        )

    if not obs_perts or not pred_perts:
        if return_details:
            return float("nan"), {}
        return float("nan")

    try:
        ordered_perts = sorted(obs_perts)
    except TypeError:
        ordered_perts = sorted(obs_perts, key=lambda x: str(x))

    obs_idx_by_pert = {pert: np.flatnonzero(obs_labels == pert) for pert in ordered_perts}
    pred_idx_by_pert = {pert: np.flatnonzero(pred_labels == pert) for pert in ordered_perts}

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

    per_pert_dists: list[float] = []
    for pert in ordered_perts:
        x_obs = extract_rows(obs, obs_idx_by_pert[pert], layer_obs)
        x_pred = extract_rows(pred, pred_idx_by_pert[pert], layer_pred)
        if x_obs.size == 0 or x_pred.size == 0:
            d = np.nan
        else:
            if pca_model is not None:
                x_obs = pca_model.transform(x_obs).astype(np.float64, copy=False)
                x_pred = pca_model.transform(x_pred).astype(np.float64, copy=False)

            if method == "parametric":
                d = param_distance(
                    x_obs=x_obs,
                    x_pred=x_pred,
                    parametric_form=distribution_form,
                    dist_type=dist_type,
                )
            elif method == "mmd":
                d = mmd_distance(
                    x_obs=x_obs,
                    x_pred=x_pred,
                    kernel=kernel,
                    **kernel_params,
                )
            elif method == "fid":
                d = fid_distance(x_obs=x_obs, x_pred=x_pred)
            else:
                raise ValueError(f"Unsupported method: {method}")

        per_pert_dists.append(d)

    if not per_pert_dists:
        if return_details:
            return float("nan"), {}
        return float("nan")
    if aggregate == "mean":
        if return_details:
            return float(np.nanmean(per_pert_dists)), {
                pert: dist for pert, dist in zip(ordered_perts, per_pert_dists, strict=True)
            }
        return float(np.nanmean(per_pert_dists))
    if return_details:
        return float(np.nanmedian(per_pert_dists)), {
            pert: dist for pert, dist in zip(ordered_perts, per_pert_dists, strict=True)
        }
    return float(np.nanmedian(per_pert_dists))
