"""Evaluation helpers for perturbation prediction quality metrics."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from ..metrics.gene_selection.differential_expression_score import (
    des_from_de_tables,
    scanpy_de_table,
)
from ..metrics.perturbation_effect.pearson import pearson_pert
from ..metrics.perturbation_effect.perturbation_discrimination_score import pds
from ..metrics.perturbation_effect.r_square import r2_score_pert
from ..metrics.reconstruction.distribution_distance import distribution_distance
from ..metrics.reconstruction.mean_error import mean_error_pert
from ..metrics.reconstruction.vendi_score import vendi_score, vendi_score_pseudobulk
from ..util.anndata_util import get_matrix, obs_has_key


def _to_vector(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


def _sort_labels(labels: set[Any]) -> np.ndarray:
    try:
        ordered = sorted(labels)
    except TypeError:
        ordered = sorted(labels, key=lambda x: str(x))
    return np.asarray(ordered)


def initial_des_scores() -> dict[str, float]:
    """Return a default dict of differential expression score metrics with NaN values."""
    return {
        "des_recall": np.nan,
        "des_precision": np.nan,
        "des_jaccard": np.nan,
    }


def _has_selected_genes(mask: np.ndarray) -> bool:
    return bool(np.any(np.asarray(mask, dtype=bool)))


def get_eval_perturbation_ids(
    obs: Any = None,
    pred: Any = None,
    control_label: str | int | float = -1,
    strict_match: bool = True,
) -> np.ndarray:
    """Return sorted non-control perturbation labels from obs/pred objects."""
    per_source: dict[str, set[Any]] = {}

    for name, data_obj in (("obs", obs), ("pred", pred)):
        if data_obj is None:
            continue
        obs_df = getattr(data_obj, "obs", None)
        if not obs_has_key(obs_df, "perturbation"):
            continue
        labels = np.asarray(data_obj.obs["perturbation"])
        per_source[name] = set(np.unique(labels)) - {control_label}

    if not per_source:
        return np.asarray([], dtype=object)

    if strict_match and len(per_source) >= 2:
        value_sets = list(per_source.values())
        reference = value_sets[0]
        for current in value_sets[1:]:
            if current != reference:
                raise ValueError(
                    "obs and pred have mismatched perturbation labels (excluding control). "
                    f"obs_only={sorted(per_source.get('obs', set()) - per_source.get('pred', set()))}, "
                    f"pred_only={sorted(per_source.get('pred', set()) - per_source.get('obs', set()))}."
                )
        return _sort_labels(reference)

    merged: set[Any] = set()
    for ids in per_source.values():
        merged.update(ids)
    return _sort_labels(merged)


def _build_mu_pred_from_prediction(
    *,
    pred: Any,
    perturbation_ids: np.ndarray,
    layer_name: str | None,
) -> np.ndarray:
    """Aggregate a prediction AnnData-like object into perturbation pseudobulks."""
    if pred is None:
        raise ValueError("pred must be provided when mu_pred is None.")
    if "perturbation" not in pred.obs.columns:
        raise KeyError("pred.obs must contain 'perturbation' when mu_pred is None.")

    mu_pred = np.empty((perturbation_ids.size, pred.shape[1]), dtype=np.float32)
    pred_labels = np.asarray(pred.obs["perturbation"])
    pred_matrix = get_matrix(pred, layer_key=layer_name)

    for idx, pert_id in enumerate(perturbation_ids):
        pert_mask = pred_labels == pert_id
        if int(np.sum(pert_mask)) == 0:
            raise ValueError(f"No cells found for perturbation label {pert_id!r} in pred.")
        mu_pred[idx, :] = _to_vector(pred_matrix[pert_mask, :].mean(axis=0)).astype(
            np.float32,
            copy=False,
        )
    return mu_pred


def _prepare_evaluation_inputs(
    *,
    pred: Any,
    obs: Any,
    mu_pred: np.ndarray | None,
    mu_obs: np.ndarray,
    true_DEGs: Any,
    obs_DEGs: list[np.ndarray],
    perturbation_ids: np.ndarray | None,
    layer_name: str | None,
    control_label: str | int | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve and validate perturbation-aligned inputs shared by evaluators."""
    resolved_perturbation_ids = (
        np.asarray(perturbation_ids)
        if perturbation_ids is not None
        else get_eval_perturbation_ids(
            obs=obs,
            pred=pred,
            control_label=control_label,
            strict_match=bool(obs is not None and pred is not None),
        )
    )

    n_perts = int(mu_obs.shape[0])
    if resolved_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbation labels found for evaluation.")
    if resolved_perturbation_ids.size != n_perts:
        raise ValueError(
            "Length mismatch: "
            f"perturbation_ids={resolved_perturbation_ids.size}, mu_obs rows={n_perts}."
        )
    if true_DEGs is not None and len(true_DEGs) != n_perts:
        raise ValueError(f"Length mismatch: true_DEGs={len(true_DEGs)}, expected {n_perts}.")
    if len(obs_DEGs) != n_perts:
        raise ValueError(f"Length mismatch: obs_DEGs={len(obs_DEGs)}, expected {n_perts}.")

    resolved_mu_pred = (
        mu_pred
        if mu_pred is not None
        else _build_mu_pred_from_prediction(
            pred=pred,
            perturbation_ids=resolved_perturbation_ids,
            layer_name=layer_name,
        )
    )
    return resolved_perturbation_ids, resolved_mu_pred


def evaluation(
    pred,
    obs,
    mu_pred,
    mu_obs,
    mu_control_obs,
    mu_pool_obs,
    true_DEGs,
    obs_DE_table,
    obs_DEGs,
    mmd_gamma: float,
    mmd_pca_model: Any | None,
    vendi_outer_sigma_squared: float,
    vendi_pseudobulk_pca_model: Any | None,
    vendi_pseudobulk_sigma_squared: float | None,
    vendi_score_obs: float | None = None,
    fdr_threshold=0.05,
    perturbation_ids: np.ndarray | None = None,
    model: str = "Average",
    layer_name: str | None = "normalized_log1p",
    control_label: str | int | float = -1,
):
    """Evaluate predicted perturbation profiles for synthetic or real datasets."""
    des_scores = initial_des_scores()

    perturbation_ids, mu_pred = _prepare_evaluation_inputs(
        pred=pred,
        obs=obs,
        mu_pred=mu_pred,
        mu_obs=mu_obs,
        true_DEGs=true_DEGs,
        obs_DEGs=obs_DEGs,
        perturbation_ids=perturbation_ids,
        layer_name=layer_name,
        control_label=control_label,
    )
    n_perts = perturbation_ids.size

    obs_has_control = bool(np.any(np.asarray(obs.obs["perturbation"]) == control_label))
    if pred is not None and obs is not None:
        # This has major computational burden (because it cannot use PCA)
        parametric_distance = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_name,
            layer_pred=layer_name,
            control_label=control_label,
            method="parametric",
            distribution_form="NB",
            dist_type="JS-divergence",
            use_pca=False,
        )

        mmd_distance = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_name,
            layer_pred=layer_name,
            control_label=control_label,
            method="mmd",
            distribution_form="NB",
            use_pca=bool(obs_has_control),
            aggregate="mean",
            pca_model=mmd_pca_model,
            gamma=mmd_gamma,
        )
        fid_distance = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_name,
            layer_pred=layer_name,
            control_label=control_label,
            method="fid",
            use_pca=bool(obs_has_control),
            aggregate="mean",
            pca_model=mmd_pca_model,
        )

        pred_has_control = bool(np.any(np.asarray(pred.obs["perturbation"]) == control_label))
        vendi_score_pred = vendi_score(
            ac=pred,
            n_pca_components=50,
            layer_key=layer_name,
            control_label=control_label if pred_has_control else None,
            gamma=mmd_gamma,
            pca_model=mmd_pca_model,
            outer_sigma_squared=vendi_outer_sigma_squared,
        )

        if pred_has_control:
            pred_for_de = pred
        elif obs_has_control:
            obs_labels = np.asarray(obs.obs["perturbation"])
            control_idx = np.flatnonzero(obs_labels == control_label)
            obs_controls = obs[control_idx, :]
            if hasattr(obs_controls, "to_adata"):
                obs_controls = obs_controls.to_adata()
            else:
                obs_controls = obs_controls.copy()
            pred_for_de = ad.concat([pred, obs_controls], join="inner", merge="same")
        else:
            pred_for_de = None

        if pred_for_de is not None:
            pred_DE_table = scanpy_de_table(
                adata=pred_for_de,
                pert_col="perturbation",
                control_pert=control_label,
                key_added="de_pred",
                layer=layer_name,
            )
            des_results = des_from_de_tables(
                real_de=obs_DE_table,
                pred_de=pred_DE_table,
                fdr_threshold=fdr_threshold,
            )
            des_scores = {
                f"des_{metric_name}": float(metric_result["overall"])
                for metric_name, metric_result in des_results.items()
            }
    else:
        # for models that only output pseudobulk
        parametric_distance = np.nan
        mmd_distance = np.nan
        fid_distance = np.nan
        if int(mu_pred.shape[0]) == n_perts + 1:
            control_idx = 0
        elif int(mu_pred.shape[0]) == n_perts:
            control_idx = None
        else:
            raise ValueError(
                f"mu_pred has unexpected number of rows ({mu_pred.shape[0]}). "
                f"Expected {n_perts} (no control row) or {n_perts + 1} (with control row)."
            )
        vendi_score_pred = vendi_score_pseudobulk(
            mu_pred,
            control_idx=control_idx,
            pca_model=vendi_pseudobulk_pca_model,
            outer_sigma_squared=vendi_pseudobulk_sigma_squared,
        )
        vendi_score_obs = vendi_score_pseudobulk(
            mu_obs,
            pca_model=vendi_pseudobulk_pca_model,
            outer_sigma_squared=vendi_pseudobulk_sigma_squared,
        )

    # Get vendi score for the observed data as well
    if vendi_score_obs is None:
        vendi_score_obs = vendi_score(
            ac=obs,
            n_pca_components=50,
            layer_key=layer_name,
            control_label=control_label if obs_has_control else None,
            gamma=mmd_gamma,
            pca_model=mmd_pca_model,
            outer_sigma_squared=vendi_outer_sigma_squared,
        )

    pds_l1_score = pds(
        X_obs=mu_obs,
        X_pred=mu_pred,
        reference=mu_control_obs,
        metric="l1",
    )
    pds_l2_score = pds(
        X_obs=mu_obs,
        X_pred=mu_pred,
        reference=mu_control_obs,
        metric="l2",
    )
    pds_cosine_score = pds(
        X_obs=mu_obs,
        X_pred=mu_pred,
        reference=mu_control_obs,
        metric="cosine",
    )

    tracker = {
        "pearson": [],
        "pearson_true_degs": [],
        "pearson_degs": [],
        "mae": [],
        "mae_true_degs": [],
        "mae_degs": [],
        "mse": [],
        "mse_true_degs": [],
        "mse_degs": [],
        "r2": [],
        "r2_true_degs": [],
        "r2_degs": [],
    }

    for idx in range(n_perts):
        obs_degs = obs_DEGs[idx]
        obs_has_selected_degs = _has_selected_genes(obs_degs)
        mu_obs_ptb = mu_obs[idx].astype(np.float32, copy=False)
        mu_pred_ptb = mu_pred[idx].astype(np.float32, copy=False)

        if model != "Control":
            tracker["pearson"].append(
                pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs)
            )
            if obs_has_selected_degs:
                tracker["pearson_degs"].append(
                    pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs, DEGs=obs_degs)
                )

        tracker["mae"].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute"))
        tracker["mse"].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared"))
        if obs_has_selected_degs:
            tracker["mae_degs"].append(
                mean_error_pert(
                    mu_obs_ptb,
                    mu_pred_ptb,
                    type="absolute",
                    weights=obs_degs.astype(np.float32),
                )
            )
            tracker["mse_degs"].append(
                mean_error_pert(
                    mu_obs_ptb,
                    mu_pred_ptb,
                    type="squared",
                    weights=obs_degs.astype(np.float32),
                )
            )
        tracker["r2"].append(r2_score_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs))
        if obs_has_selected_degs:
            tracker["r2_degs"].append(
                r2_score_pert(
                    mu_obs_ptb,
                    mu_pred_ptb,
                    reference=mu_control_obs,
                    weights=obs_degs.astype(np.float32),
                )
            )

        if true_DEGs is not None:
            true_degs = true_DEGs[idx]
            true_has_selected_degs = _has_selected_genes(true_degs)
            if true_has_selected_degs:
                tracker["pearson_true_degs"].append(
                    pearson_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        reference=mu_control_obs,
                        DEGs=true_degs,
                    )
                )
                tracker["mae_true_degs"].append(
                    mean_error_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        type="absolute",
                        weights=true_degs.astype(np.float32),
                    )
                )
                tracker["mse_true_degs"].append(
                    mean_error_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        type="squared",
                        weights=true_degs.astype(np.float32),
                    )
                )
                tracker["r2_true_degs"].append(
                    r2_score_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        reference=mu_control_obs,
                        weights=true_degs.astype(np.float32),
                    )
                )

    return {
        "pearson": np.nanmedian(tracker["pearson"]) if tracker["pearson"] else np.nan,
        "pearson_true_degs": (
            np.nanmedian(tracker["pearson_true_degs"]) if tracker["pearson_true_degs"] else np.nan
        ),
        "pearson_degs": (
            np.nanmedian(tracker["pearson_degs"]) if tracker["pearson_degs"] else np.nan
        ),
        "mae": np.nanmedian(tracker["mae"]) if tracker["mae"] else np.nan,
        "mae_true_degs": (
            np.nanmedian(tracker["mae_true_degs"]) if tracker["mae_true_degs"] else np.nan
        ),
        "mae_degs": (np.nanmedian(tracker["mae_degs"]) if tracker["mae_degs"] else np.nan),
        "mse": np.nanmedian(tracker["mse"]) if tracker["mse"] else np.nan,
        "mse_true_degs": (
            np.nanmedian(tracker["mse_true_degs"]) if tracker["mse_true_degs"] else np.nan
        ),
        "mse_degs": (np.nanmedian(tracker["mse_degs"]) if tracker["mse_degs"] else np.nan),
        "r2": np.nanmedian(tracker["r2"]) if tracker["r2"] else np.nan,
        "r2_true_degs": (
            np.nanmedian(tracker["r2_true_degs"]) if tracker["r2_true_degs"] else np.nan
        ),
        "r2_degs": np.nanmedian(tracker["r2_degs"]) if tracker["r2_degs"] else np.nan,
        "parametric_distance": parametric_distance,
        "mmd_distance": mmd_distance,
        "fid_distance": fid_distance,
        "vendi_score_pred": vendi_score_pred,
        "vendi_score_obs": vendi_score_obs,
        **des_scores,
        "pds_l1": pds_l1_score,
        "pds_l2": pds_l2_score,
        "pds_cosine": pds_cosine_score,
    }


def evaluation_by_perturbation(
    pred,
    obs,
    mu_pred,
    mu_obs,
    mu_control_obs,
    true_DEGs,
    obs_DEGs,
    mmd_gamma: float,
    mmd_pca_model: Any | None,
    perturbation_ids: np.ndarray | None = None,
    model: str = "Average",
    layer_name: str | None = "normalized_log1p",
    control_label: str | int | float = -1,
) -> list[dict[str, Any]]:
    """Evaluate predicted perturbation profiles and return one metrics row per perturbation."""
    perturbation_ids, mu_pred = _prepare_evaluation_inputs(
        pred=pred,
        obs=obs,
        mu_pred=mu_pred,
        mu_obs=mu_obs,
        true_DEGs=true_DEGs,
        obs_DEGs=obs_DEGs,
        perturbation_ids=perturbation_ids,
        layer_name=layer_name,
        control_label=control_label,
    )
    n_perts = perturbation_ids.size

    obs_has_control = bool(np.any(np.asarray(obs.obs["perturbation"]) == control_label))
    if pred is not None and obs is not None:
        _, parametric_distance_by_pert = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_name,
            layer_pred=layer_name,
            control_label=control_label,
            method="parametric",
            distribution_form="NB",
            dist_type="JS-divergence",
            use_pca=False,
            return_details=True,
        )

        _, mmd_distance_by_pert = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_name,
            layer_pred=layer_name,
            control_label=control_label,
            method="mmd",
            distribution_form="NB",
            use_pca=bool(obs_has_control),
            aggregate="mean",
            pca_model=mmd_pca_model,
            gamma=mmd_gamma,
            return_details=True,
        )
        _, fid_distance_by_pert = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_name,
            layer_pred=layer_name,
            control_label=control_label,
            method="fid",
            use_pca=bool(obs_has_control),
            aggregate="mean",
            pca_model=mmd_pca_model,
            return_details=True,
        )

    else:
        parametric_distance_by_pert = {}
        mmd_distance_by_pert = {}
        fid_distance_by_pert = {}

    rows: list[dict[str, Any]] = []
    for idx in range(n_perts):
        perturbation_id = perturbation_ids[idx]
        obs_degs = np.asarray(obs_DEGs[idx], dtype=bool)
        obs_has_selected_degs = _has_selected_genes(obs_degs)
        mu_obs_ptb = mu_obs[idx].astype(np.float32, copy=False)
        mu_pred_ptb = mu_pred[idx].astype(np.float32, copy=False)
        row = {
            "perturbation": str(perturbation_id),
            "n_obs_degs": int(np.sum(obs_degs)),
            "n_obs_degs_in_truth": np.nan,
            "n_deg_truth": np.nan,
            "pearson": np.nan,
            "pearson_degs": np.nan,
            "pearson_true_degs": np.nan,
            "mae": float(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute")),
            "mae_degs": np.nan,
            "mae_true_degs": np.nan,
            "mse": float(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared")),
            "mse_degs": np.nan,
            "mse_true_degs": np.nan,
            "r2": float(r2_score_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs)),
            "r2_degs": np.nan,
            "r2_true_degs": np.nan,
            "parametric_distance": float(parametric_distance_by_pert.get(perturbation_id, np.nan)),
            "mmd_distance": float(mmd_distance_by_pert.get(perturbation_id, np.nan)),
            "fid_distance": float(fid_distance_by_pert.get(perturbation_id, np.nan)),
        }

        if model != "Control":
            row["pearson"] = float(
                pearson_pert(
                    mu_obs_ptb,
                    mu_pred_ptb,
                    reference=mu_control_obs,
                )
            )
            if obs_has_selected_degs:
                row["pearson_degs"] = float(
                    pearson_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        reference=mu_control_obs,
                        DEGs=obs_degs,
                    )
                )

        if obs_has_selected_degs:
            weights = obs_degs.astype(np.float32)
            row["mae_degs"] = float(
                mean_error_pert(
                    mu_obs_ptb,
                    mu_pred_ptb,
                    type="absolute",
                    weights=weights,
                )
            )
            row["mse_degs"] = float(
                mean_error_pert(
                    mu_obs_ptb,
                    mu_pred_ptb,
                    type="squared",
                    weights=weights,
                )
            )
            row["r2_degs"] = float(
                r2_score_pert(
                    mu_obs_ptb,
                    mu_pred_ptb,
                    reference=mu_control_obs,
                    weights=weights,
                )
            )

        if true_DEGs is not None:
            true_degs = np.asarray(true_DEGs[idx], dtype=bool)
            row["n_obs_degs_in_truth"] = int(np.sum(obs_degs & true_degs))
            row["n_deg_truth"] = int(np.sum(true_degs))

            if _has_selected_genes(true_degs):
                true_weights = true_degs.astype(np.float32)
                row["pearson_true_degs"] = float(
                    pearson_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        reference=mu_control_obs,
                        DEGs=true_degs,
                    )
                )
                row["mae_true_degs"] = float(
                    mean_error_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        type="absolute",
                        weights=true_weights,
                    )
                )
                row["mse_true_degs"] = float(
                    mean_error_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        type="squared",
                        weights=true_weights,
                    )
                )
                row["r2_true_degs"] = float(
                    r2_score_pert(
                        mu_obs_ptb,
                        mu_pred_ptb,
                        reference=mu_control_obs,
                        weights=true_weights,
                    )
                )

        rows.append(row)

    return rows
