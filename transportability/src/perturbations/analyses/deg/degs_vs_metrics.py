"""Export held-out per-perturbation DEG counts and evaluation metrics."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ...data.dgp import directDGP
from ...metrics.gene_selection.differential_expression_score import (
    de_table_to_deg_masks,
    scanpy_de_table,
)
from ...metrics.reconstruction.distance_util import estimate_mmd_gamma
from ...models.linear import predict_linear_pca_baseline
from ...models.optional import ScviPerturbation, run_cpa, run_gears, run_state_gene
from ...util.anndata_util import fit_control_incremental_pca, get_matrix
from ..common import MODELS, NORM_LAYER_KEY, label_to_target_tokens
from ..context import DEFAULT_CONTEXT_AXIS, ContextSplitter
from ..evaluator import evaluation_by_perturbation, get_eval_perturbation_ids
from ..synthetic_simulations.sampling import (
    PARAM_RANGES,
    load_parameter_estimation_inputs,
    sample_parameters,
)
from ..util import (
    build_perturbation_id_map,
    compute_means_by_perturbation,
    count_non_control_perturbations,
    ensure_normalized_log1p_layer,
    load_real_dataset,
    true_degs_for_context,
    validate_perturbation_targets_subset,
)

_OUTPUT_DIR = Path("results/deg_vs_metrics")
_SUPPORTED_DATASETS = ("norman19", "directDGP")
_CONTROL_LABEL = "control"
_CONTROL_ID = np.asarray([_CONTROL_LABEL], dtype=object)
_DIRECT_DGP_PARAM_COLUMNS = [
    "G",
    "N0",
    "Nk",
    "P",
    "p_effect",
    "effect_factor",
    "B",
    "mu_l",
]
_ROW_METRIC_COLUMNS = [
    "perturbation",
    "n_obs_degs",
    "n_obs_degs_in_truth",
    "n_deg_truth",
    "pearson",
    "pearson_degs",
    "pearson_true_degs",
    "mae",
    "mae_degs",
    "mae_true_degs",
    "mse",
    "mse_degs",
    "mse_true_degs",
    "r2",
    "r2_degs",
    "r2_true_degs",
    "parametric_distance",
    "mmd_distance",
    "fid_distance",
]
_RESULT_COLUMNS = [
    "dataset",
    "dataset_path",
    "split_strategy",
    "n_cells",
    "n_genes",
    "n_total_perturbations",
    "seed",
    *_DIRECT_DGP_PARAM_COLUMNS,
    "model",
    "trial_id",
    "trial_seed",
    "status",
    "error",
    "execution_time",
    "context_axis",
    "context_values",
    *_ROW_METRIC_COLUMNS,
]


@dataclass(frozen=True)
class PreparedTrialData:
    """Precomputed state shared across models for one held-out trial."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_adata: ad.AnnData
    val_adata: ad.AnnData
    test_adata: ad.AnnData
    obs_eval: ad.AnnData
    perturbation_ids: np.ndarray
    mu_obs: np.ndarray
    mu_control_obs: np.ndarray
    mu_train: np.ndarray
    mu_control_train: np.ndarray
    mu_pool_train: np.ndarray
    train_targets: list[tuple[str, ...]]
    test_targets: list[tuple[str, ...]]
    gene_names: np.ndarray
    obs_degs: list[np.ndarray]
    true_degs: list[np.ndarray] | None
    context_values: str
    mmd_gamma: float
    mmd_pca_model: Any | None


def _filter_evaluation_bucket_min_cells(
    adata_obj: ad.AnnData,
    *,
    perturbation_key: str = "perturbation",
    control_label: str = _CONTROL_LABEL,
    min_cells_per_non_control: int = 2,
    min_cells_for_control: int = 2,
) -> ad.AnnData:
    """Drop held-out perturbation groups that are too small for DE analysis."""
    labels = adata_obj.obs[perturbation_key].astype(str)
    counts = labels.value_counts(sort=False)

    if int(counts.get(control_label, 0)) < int(min_cells_for_control):
        return adata_obj[[], :].copy()

    dropped = sorted(
        label
        for label, count in counts.items()
        if label != control_label and int(count) < int(min_cells_per_non_control)
    )
    if not dropped:
        return adata_obj

    keep_mask = ~labels.isin(dropped)
    return adata_obj[keep_mask.to_numpy(), :].copy()


def _make_prediction_adata(
    *,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    prediction_matrix: Any,
    layer_name: str | None,
) -> ad.AnnData:
    """Build a lightweight prediction AnnData aligned to held-out test metadata."""
    copied_obs = obs.copy()
    copied_var = var.copy()
    if sparse.issparse(prediction_matrix):
        values = prediction_matrix.copy().astype(np.float32, copy=False)
    else:
        values = np.asarray(prediction_matrix, dtype=np.float32).copy()
    if layer_name is None:
        return ad.AnnData(X=values, obs=copied_obs, var=copied_var)

    pred = ad.AnnData(
        X=sparse.csr_matrix((copied_obs.shape[0], copied_var.shape[0]), dtype=np.float32),
        obs=copied_obs,
        var=copied_var,
    )
    pred.layers[layer_name] = values
    return pred


def _filter_prediction_to_eval_perturbations(
    pred: ad.AnnData,
    perturbation_ids: np.ndarray,
    *,
    control_label: str = _CONTROL_LABEL,
) -> ad.AnnData:
    """Restrict predictions to evaluated perturbations plus controls."""
    allowed = np.concatenate(
        [
            np.asarray([control_label], dtype=object),
            np.asarray(perturbation_ids, dtype=object),
        ]
    )
    keep_mask = np.isin(
        pred.obs["perturbation"].to_numpy(copy=False),
        allowed,
    )
    return pred[keep_mask, :].copy()


def _trial_model_dir(backend_name: str, pid: int, trial_id: int) -> Path:
    """Return the on-disk output directory for one learned-model trial run."""
    model_dir = _OUTPUT_DIR / backend_name / str(pid) / f"trial_{trial_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _empty_metric_values() -> dict[str, Any]:
    """Return one blank per-perturbation metric payload."""
    metrics = {column: np.nan for column in _ROW_METRIC_COLUMNS}
    metrics["perturbation"] = ""
    return metrics


def _failure_rows(
    *,
    base_row: dict[str, Any],
    models: tuple[str, ...],
    error: str,
    context_values: str = "",
) -> list[dict[str, Any]]:
    """Build failure rows for one or more models."""
    return [
        {
            **base_row,
            "model": model,
            "status": "failed",
            "error": error,
            "execution_time": np.nan,
            "context_axis": DEFAULT_CONTEXT_AXIS,
            "context_values": context_values,
            **_empty_metric_values(),
        }
        for model in models
    ]


def _prepare_real_dataset(
    *,
    dataset_path: str,
    counts_layer: str | None,
    obs_layer: str | None,
    norm_target_sum: float,
) -> ad.AnnData:
    """Load Norman19 and ensure the layers required by evaluation are present."""
    adata, _ = load_real_dataset(dataset_path=dataset_path)

    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(
            f"Requested counts_layer={counts_layer!r} not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    source_layer = counts_layer or ("counts" if "counts" in adata.layers else None)
    ensure_normalized_log1p_layer(
        adata=adata,
        output_layer_key=NORM_LAYER_KEY,
        source_layer=source_layer,
        target_sum=norm_target_sum,
    )

    if obs_layer is not None and obs_layer not in adata.layers:
        raise KeyError(
            f"Requested obs_layer={obs_layer!r} not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    validate_perturbation_targets_subset(adata=adata, control_label=_CONTROL_LABEL)
    return adata


def _prepare_direct_dgp_trial(
    *,
    rng: np.random.Generator,
    synthetic_inputs: dict[str, np.ndarray],
    trial_seed: int,
    obs_layer: str | None,
) -> tuple[ad.AnnData, list[np.ndarray], dict[str, Any]]:
    """Sample one directDGP dataset and return it with sampled parameters."""
    sampled_params = sample_parameters(PARAM_RANGES, rng)
    adata, affected_genes = directDGP(
        G=int(sampled_params["G"]),
        N0=int(sampled_params["N0"]),
        Nk=int(sampled_params["Nk"]),
        P=int(sampled_params["P"]),
        p_effect=float(sampled_params["p_effect"]),
        effect_factor=float(sampled_params["effect_factor"]),
        B=float(sampled_params["B"]),
        mu_l=float(sampled_params["mu_l"]),
        all_theta=synthetic_inputs["all_theta"],
        control_mu=synthetic_inputs["control_mu"],
        pert_mu=synthetic_inputs["pert_mu"],
        gene_names=synthetic_inputs["gene_names"],
        seed=trial_seed,
        normalize=True,
        normalized_layer_key=NORM_LAYER_KEY,
    )
    if obs_layer is not None and obs_layer not in adata.layers:
        raise KeyError(
            f"Requested obs_layer={obs_layer!r} not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    return (
        adata,
        affected_genes,
        {column: sampled_params[column] for column in _DIRECT_DGP_PARAM_COLUMNS},
    )


def _prepare_trial_data(
    *,
    dataset_name: str,
    adata: ad.AnnData,
    trial_seed: int,
    obs_layer: str | None,
    affected_genes: list[np.ndarray] | None,
) -> PreparedTrialData:
    """Prepare split-aligned DEG masks and mean profiles for one trial."""
    splitter = ContextSplitter(
        adata=adata,
        split_strategy="in-context",
        context_axis=DEFAULT_CONTEXT_AXIS,
        control_label=_CONTROL_LABEL,
    )
    train_idx, val_idx, test_idx = splitter.split(seed=trial_seed)
    split_metadata = splitter.get_split_metadata(seed=trial_seed)
    if len(split_metadata.evaluation_contexts) != 1:
        raise ValueError(
            "degs_vs_metrics expects exactly one evaluation context for supported datasets."
        )
    eval_context = split_metadata.evaluation_contexts[0]

    train_adata = adata[train_idx, :].copy()
    val_adata = adata[val_idx, :].copy()
    test_adata = adata[test_idx, :].copy()
    obs_eval = _filter_evaluation_bucket_min_cells(test_adata)

    perturbation_ids = get_eval_perturbation_ids(
        obs=obs_eval,
        control_label=_CONTROL_LABEL,
        strict_match=False,
    )
    if perturbation_ids.size == 0:
        raise ValueError(
            "No held-out perturbations remained after filtering test groups with fewer than 2 cells."
        )

    train_val_adata = train_adata
    if val_adata.n_obs > 0:
        train_val_adata = ad.concat(
            [train_adata, val_adata],
            join="inner",
            merge="same",
            uns_merge="same",
            index_unique=None,
        )

    train_perturbation_ids = get_eval_perturbation_ids(
        obs=train_val_adata,
        control_label=_CONTROL_LABEL,
        strict_match=False,
    )
    if train_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbations found in the train/validation split.")

    mu_train = compute_means_by_perturbation(
        adata_view=train_val_adata,
        perturbation_ids=train_perturbation_ids,
        layer_key=obs_layer,
        missing_group_context="train/validation split",
    )
    mu_control_train = compute_means_by_perturbation(
        adata_view=train_val_adata,
        perturbation_ids=_CONTROL_ID,
        layer_key=obs_layer,
        missing_group_context="train/validation split",
    )[0]
    train_labels = train_val_adata.obs["perturbation"].to_numpy(copy=False)
    train_counts = np.array(
        [np.sum(train_labels == pert_id) for pert_id in train_perturbation_ids],
        dtype=np.float64,
    )
    mu_pool_train = np.average(mu_train, axis=0, weights=train_counts).astype(
        np.float32,
        copy=False,
    )

    mu_obs = compute_means_by_perturbation(
        adata_view=obs_eval,
        perturbation_ids=perturbation_ids,
        layer_key=obs_layer,
        missing_group_context="held-out evaluation split",
    )
    mu_control_obs = compute_means_by_perturbation(
        adata_view=obs_eval,
        perturbation_ids=_CONTROL_ID,
        layer_key=obs_layer,
        missing_group_context="held-out evaluation split",
    )[0]
    de_table = scanpy_de_table(
        adata=obs_eval.copy(),
        pert_col="perturbation",
        control_pert=_CONTROL_LABEL,
        key_added="test_de",
        layer=obs_layer,
    )
    obs_degs = de_table_to_deg_masks(
        de_table=de_table,
        gene_names=np.asarray(obs_eval.var_names, dtype=str),
        perturbation_ids=perturbation_ids,
        fdr_threshold=0.05,
    )

    mmd_pca_model = fit_control_incremental_pca(
        data_obj=obs_eval,
        layer_key=obs_layer,
        control_label=_CONTROL_LABEL,
        obs_key="perturbation",
        data_name="obs_eval",
    )
    mmd_gamma = estimate_mmd_gamma(
        obs=obs_eval,
        layer_obs=obs_layer,
        control_label=_CONTROL_LABEL,
        pca_model=mmd_pca_model,
    )

    true_degs = None
    if dataset_name == "directDGP":
        if affected_genes is None:
            raise ValueError("directDGP trial preparation requires affected_genes.")
        true_degs = true_degs_for_context(
            dataset_name=dataset_name,
            affected_genes=affected_genes,
            pert_label_to_id=build_perturbation_id_map(adata.obs),
            perturbation_ids=perturbation_ids,
            context_axis=eval_context.axis,
            context_values=eval_context.values,
        )

    return PreparedTrialData(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_adata=train_adata,
        val_adata=val_adata,
        test_adata=test_adata,
        obs_eval=obs_eval,
        perturbation_ids=perturbation_ids,
        mu_obs=mu_obs,
        mu_control_obs=mu_control_obs,
        mu_train=mu_train,
        mu_control_train=mu_control_train,
        mu_pool_train=mu_pool_train,
        train_targets=[label_to_target_tokens(str(pid)) for pid in train_perturbation_ids],
        test_targets=[label_to_target_tokens(str(pid)) for pid in perturbation_ids],
        gene_names=np.asarray(train_val_adata.var_names, dtype=str),
        obs_degs=obs_degs,
        true_degs=true_degs,
        context_values=eval_context.value_label,
        mmd_gamma=mmd_gamma,
        mmd_pca_model=mmd_pca_model,
    )


def _baseline_prediction(
    *,
    model: str,
    prepared: PreparedTrialData,
    trial_seed: int,
) -> np.ndarray:
    """Return per-perturbation pseudobulk predictions for baseline models."""
    if model == "Control":
        return np.tile(prepared.mu_control_train, (prepared.perturbation_ids.size, 1))
    if model in {"Average", "Context-Average"}:
        return np.tile(prepared.mu_pool_train, (prepared.perturbation_ids.size, 1))
    if model in {"linearPCA", "Context-linearPCA"}:
        return predict_linear_pca_baseline(
            train_means=prepared.mu_train,
            train_target_genes=prepared.train_targets,
            test_target_genes=prepared.test_targets,
            gene_names=prepared.gene_names,
            fallback_mean=prepared.mu_pool_train,
            seed=trial_seed,
        )
    raise NotImplementedError(f"Model {model!r} is not implemented.")


def _learned_prediction(
    *,
    dataset_name: str,
    adata: ad.AnnData,
    prepared: PreparedTrialData,
    model: str,
    trial_id: int,
    trial_seed: int,
    pid: int,
    counts_layer: str | None,
    obs_layer: str | None,
    norm_target_sum: float,
) -> ad.AnnData:
    """Run one cell-level model and return held-out predictions as AnnData."""
    trial_dataset_name = f"{dataset_name}_trial_{trial_id}"
    is_norman19 = dataset_name == "norman19"
    model_input_layer = obs_layer if is_norman19 else NORM_LAYER_KEY
    normalized_target_sum = norm_target_sum if is_norman19 else 1e4

    def prediction_from_test_matrix(matrix: Any) -> ad.AnnData:
        pred = _make_prediction_adata(
            obs=prepared.test_adata.obs,
            var=prepared.test_adata.var,
            prediction_matrix=matrix,
            layer_name=obs_layer,
        )
        return _filter_prediction_to_eval_perturbations(pred, prepared.perturbation_ids)

    if model == "scVI":
        scvi_model = ScviPerturbation(
            data=adata,
            counts_layer=counts_layer if is_norman19 else None,
            perturbation_key="perturbation",
            context_key=DEFAULT_CONTEXT_AXIS,
            train_idx=prepared.train_idx,
            val_idx=prepared.val_idx,
            test_idx=prepared.test_idx,
            seed=trial_seed,
        )
        scvi_pred = scvi_model.run(
            n_latent=10,
            n_hidden=128,
            n_layers=3,
            gene_likelihood="zinb",
            dispersion="gene",
            max_epochs=800,
            batch_size=512,
            early_stopping=bool(prepared.val_idx.size > 0) if is_norman19 else True,
            dataloader_num_workers=0,
            normalized_target_sum=normalized_target_sum,
        )
        pred = _make_prediction_adata(
            obs=scvi_pred.obs,
            var=scvi_pred.var,
            prediction_matrix=get_matrix(scvi_pred, layer_key=NORM_LAYER_KEY),
            layer_name=obs_layer,
        )
        return _filter_prediction_to_eval_perturbations(pred, prepared.perturbation_ids)

    if model == "GEARS":
        gears_out = run_gears(
            train_adata=prepared.train_adata,
            valid_adata=prepared.val_adata,
            test_adata=prepared.test_adata,
            is_synthetic=dataset_name == "directDGP",
            dataset_name=trial_dataset_name,
            model_dir=str(_trial_model_dir("gears", pid, trial_id)),
            perturbation_column="perturbation",
            control_label=_CONTROL_LABEL,
            expression_layer=model_input_layer,
            use_gene_ontology_graph=False,
            normalized_target_sum=normalized_target_sum,
        )
        return prediction_from_test_matrix(
            gears_out["preds"].detach().cpu().numpy().astype(np.float32, copy=False),
        )

    if model == "CPA":
        cpa_out = run_cpa(
            train_adata=prepared.train_adata,
            valid_adata=prepared.val_adata,
            test_adata=prepared.test_adata,
            add_controls=True,
            perturbation_column="perturbation",
            control_label=_CONTROL_LABEL,
            use_counts=False,
            epochs=100,
            model_dir=str(_trial_model_dir("CPA", pid, trial_id)),
            dataset_name=trial_dataset_name,
            expression_layer=model_input_layer,
            covariate_keys=[DEFAULT_CONTEXT_AXIS],
            library_size=None,
        )
        return prediction_from_test_matrix(np.asarray(cpa_out["preds"], dtype=np.float32))

    if model == "STATE":
        state_out = run_state_gene(
            train_adata=prepared.train_adata,
            valid_adata=prepared.val_adata,
            test_adata=prepared.test_adata,
            context_key=DEFAULT_CONTEXT_AXIS,
            dataset_name=trial_dataset_name,
            model_dir=str(_trial_model_dir("state_gene", pid, trial_id)),
            perturbation_column="perturbation",
            control_label=_CONTROL_LABEL,
            expression_layer=NORM_LAYER_KEY,
            epochs=100,
        )
        return prediction_from_test_matrix(np.asarray(state_out["preds"], dtype=np.float32))

    raise NotImplementedError(f"Model {model!r} is not implemented.")


def _evaluate_model_rows(
    *,
    base_row: dict[str, Any],
    prepared: PreparedTrialData,
    model: str,
    execution_time: float,
    obs_layer: str | None,
    mu_pred: np.ndarray | None,
    pred: ad.AnnData | None,
) -> list[dict[str, Any]]:
    """Evaluate one model and attach shared metadata to each perturbation row."""
    rows = evaluation_by_perturbation(
        pred=pred,
        obs=prepared.obs_eval,
        mu_pred=mu_pred,
        mu_obs=prepared.mu_obs,
        mu_control_obs=prepared.mu_control_obs,
        true_DEGs=prepared.true_degs,
        obs_DEGs=prepared.obs_degs,
        mmd_gamma=prepared.mmd_gamma,
        mmd_pca_model=prepared.mmd_pca_model,
        perturbation_ids=prepared.perturbation_ids,
        model=model,
        layer_name=obs_layer,
        control_label=_CONTROL_LABEL,
    )
    return [
        {
            **base_row,
            "model": model,
            "status": "success",
            "error": "",
            "execution_time": execution_time,
            "context_axis": DEFAULT_CONTEXT_AXIS,
            "context_values": prepared.context_values,
            **row,
        }
        for row in rows
    ]


def _write_results(rows: list[dict[str, Any]], dataset_name: str) -> Path:
    """Write the result CSV and return its path."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = _OUTPUT_DIR / f"results_{dataset_name}_{timestamp}.csv"
    frame = pd.DataFrame(rows).reindex(columns=_RESULT_COLUMNS)
    frame.to_csv(output_path, index=False)
    return output_path


def _write_error_log(rows: list[dict[str, Any]], dataset_name: str) -> Path | None:
    """Write a compact error log for failed rows, if any."""
    failed = pd.DataFrame([row for row in rows if row.get("status") == "failed"])
    if failed.empty:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_path = _OUTPUT_DIR / f"errors_{dataset_name}_{timestamp}.csv"
    failed.reindex(
        columns=[
            "trial_id",
            "trial_seed",
            "model",
            "context_values",
            "error",
        ]
    ).to_csv(error_path, index=False)
    return error_path


def run_degs_vs_metrics(
    *,
    dataset_name: str,
    dataset_path: str | None,
    n_trials: int,
    counts_layer: str | None,
    obs_layer: str | None,
    norm_target_sum: float,
    seed: int,
) -> tuple[Path, Path | None]:
    """Run per-perturbation DEG-vs-metrics export for one supported dataset."""
    if dataset_name not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}. Expected one of {_SUPPORTED_DATASETS}."
        )
    if dataset_name == "norman19" and not dataset_path:
        raise ValueError("dataset_path is required for dataset_name='norman19'.")

    pid = os.getpid()
    results: list[dict[str, Any]] = []

    base_adata: ad.AnnData | None = None
    synthetic_inputs: dict[str, np.ndarray] | None = None
    if dataset_name == "norman19":
        base_adata = _prepare_real_dataset(
            dataset_path=str(dataset_path),
            counts_layer=counts_layer,
            obs_layer=obs_layer,
            norm_target_sum=norm_target_sum,
        )
    else:
        synthetic_inputs = load_parameter_estimation_inputs()

    for trial_id in range(int(n_trials)):
        trial_seed = int(seed) + int(trial_id)
        sampled_param_row = {column: np.nan for column in _DIRECT_DGP_PARAM_COLUMNS}
        trial_base_row = {
            "dataset": dataset_name,
            "dataset_path": dataset_path or "",
            "split_strategy": "in-context",
            "n_cells": np.nan,
            "n_genes": np.nan,
            "n_total_perturbations": np.nan,
            "seed": int(seed),
            **sampled_param_row,
            "trial_id": int(trial_id),
            "trial_seed": trial_seed,
        }
        try:
            if dataset_name == "norman19":
                assert base_adata is not None
                adata = base_adata
                affected_genes = None
            else:
                assert synthetic_inputs is not None
                adata, affected_genes, sampled_param_row = _prepare_direct_dgp_trial(
                    rng=np.random.default_rng(trial_seed),
                    synthetic_inputs=synthetic_inputs,
                    trial_seed=trial_seed,
                    obs_layer=obs_layer,
                )
                trial_base_row.update(sampled_param_row)

            trial_base_row.update(
                {
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "n_total_perturbations": count_non_control_perturbations(adata.obs),
                }
            )

            prepared = _prepare_trial_data(
                dataset_name=dataset_name,
                adata=adata,
                trial_seed=trial_seed,
                obs_layer=obs_layer,
                affected_genes=affected_genes,
            )

        except Exception as exc:
            results.extend(
                _failure_rows(
                    base_row=trial_base_row,
                    models=tuple(MODELS),
                    error=str(exc),
                )
            )
            continue

        for model in MODELS:
            start_time = time.time()
            try:
                if model in {
                    "Control",
                    "Average",
                    "Context-Average",
                    "linearPCA",
                    "Context-linearPCA",
                }:
                    mu_pred = _baseline_prediction(
                        model=model,
                        prepared=prepared,
                        trial_seed=trial_seed,
                    )
                    pred = None
                elif model in {"scVI", "GEARS", "CPA", "STATE"}:
                    mu_pred = None
                    pred = _learned_prediction(
                        dataset_name=dataset_name,
                        adata=adata,
                        prepared=prepared,
                        model=model,
                        trial_id=trial_id,
                        trial_seed=trial_seed,
                        pid=pid,
                        counts_layer=counts_layer,
                        obs_layer=obs_layer,
                        norm_target_sum=norm_target_sum,
                    )
                else:
                    raise NotImplementedError(f"Model {model!r} is not implemented.")

                results.extend(
                    _evaluate_model_rows(
                        base_row=trial_base_row,
                        prepared=prepared,
                        model=model,
                        execution_time=time.time() - start_time,
                        obs_layer=obs_layer,
                        mu_pred=mu_pred,
                        pred=pred,
                    )
                )
            except Exception as exc:
                results.extend(
                    _failure_rows(
                        base_row=trial_base_row,
                        models=(model,),
                        error=str(exc),
                        context_values=prepared.context_values,
                    )
                )

    result_path = _write_results(results, dataset_name=dataset_name)
    error_path = _write_error_log(results, dataset_name=dataset_name)
    return result_path, error_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the DEG-vs-metrics export script."""
    parser = argparse.ArgumentParser(
        description="Export per-perturbation DEG counts and evaluation metrics on held-out test data."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=_SUPPORTED_DATASETS,
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--counts_layer",
        type=str,
        default="counts",
    )
    parser.add_argument(
        "--obs_layer",
        type=str,
        default=NORM_LAYER_KEY,
    )
    parser.add_argument(
        "--norm_target_sum",
        type=float,
        default=1e4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_arg_parser().parse_args()


def main() -> None:
    """Run the DEG-vs-metrics export entrypoint."""
    args = parse_args()
    result_path, error_path = run_degs_vs_metrics(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        n_trials=args.n_trials,
        counts_layer=args.counts_layer,
        obs_layer=args.obs_layer,
        norm_target_sum=args.norm_target_sum,
        seed=args.seed,
    )
    print(f"Saved results to {result_path}")
    if error_path is not None:
        print(f"Saved error log to {error_path}")


if __name__ == "__main__":
    main()
