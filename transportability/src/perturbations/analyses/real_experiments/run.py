"""Run real-experiment perturbation benchmark trials and persist evaluation outputs."""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ...data.cd4_chunked import CD4ChunkedDataset
from ...metrics.gene_selection.differential_expression_score import (
    de_table_to_deg_masks,
    scanpy_de_table,
)
from ...metrics.reconstruction.distance_util import estimate_mmd_gamma
from ...metrics.reconstruction.vendi_score import (
    estimate_vendi_outer_sigma_squared,
    estimate_vendi_pseudobulk_sigma_squared,
    fit_vendi_pseudobulk_pca,
)
from ...models.linear import predict_linear_pca_baseline
from ...models.optional import (
    ScviPerturbation,
    run_cpa,
    run_gears,
    run_scldm,
    run_state_gene,
)
from ...util.anndata_util import fit_control_incremental_pca, get_matrix
from ..common import MODELS, NORM_LAYER_KEY, label_to_target_tokens
from ..context import (
    ContextSplitter,
    build_evaluation_contexts,
    get_dataset_context_config,
    indexer_from_labels,
)
from ..evaluator import evaluation, get_eval_perturbation_ids
from ..util import (
    build_perturbation_label_mapping,
    compute_means_by_perturbation,
    ensure_normalized_log1p_layer,
    load_real_dataset,
    validate_perturbation_targets_subset_from_obs,
)

_DEFAULT_OUTPUT_DIR = "results/real_experiments"
_DEFAULT_DATASET_PATH = "data/norman19/norman19_processed.h5ad"
_CD4_SPLIT_CACHE_DIRNAME = "cd4_split_cache"
_REPLOGLE22_VARIANT_PATHS = {
    "RPE1": "data/replogle22/RPE1/processed.h5ad",
    "Jurkat": "data/replogle22/Jurkat/processed.h5ad",
    "HepG2": "data/replogle22/HepG2/processed.h5ad",
}
_METRIC_COLUMNS = [
    "pearson",
    "pearson_degs",
    "mae",
    "mae_degs",
    "mse",
    "mse_degs",
    "r2",
    "r2_degs",
    "parametric_distance",
    "mmd_distance",
    "fid_distance",
    "vendi_score_pred",
    "vendi_score_obs",
    "des_recall",
    "des_precision",
    "des_jaccard",
    "pds_l1",
    "pds_l2",
    "pds_cosine",
]


class _CudaModule(Protocol):
    """Subset of ``torch.cuda`` used during process cleanup."""

    def is_available(self) -> bool:
        """Return whether a CUDA device is available."""
        ...

    def empty_cache(self) -> None:
        """Release unoccupied cached CUDA memory."""
        ...


@dataclass(frozen=True)
class DatasetSummary:
    """Dataset-level fields copied onto every result row."""

    dataset: str
    dataset_variant: str | None
    dataset_path: str
    n_cells: int
    n_genes: int
    n_cell_lines: int
    n_total_perturbations: int
    sparsity: float


@dataclass(frozen=True)
class PreparedTrialData:
    """Precomputed evaluation state shared across model branches for one trial."""

    split_metadata: Any
    test_obs: pd.DataFrame
    test_var: pd.DataFrame
    test_perturbation_ids: np.ndarray
    mu_train: np.ndarray
    mu_control_train: np.ndarray
    mu_pool_train: np.ndarray
    train_targets: list[tuple[str, ...]]
    test_targets: list[tuple[str, ...]]
    gene_names: np.ndarray
    eval_specs: list[dict[str, Any]]


def _resolve_dataset_request(
    *,
    dataset_name: str,
    dataset_variant: str | None,
    dataset_path: str | None,
) -> tuple[str, str | None]:
    """Validate dataset selection and resolve its input path."""
    if dataset_name == "replogle22":
        if dataset_variant is None:
            raise ValueError(
                "dataset_variant is required for dataset_name='replogle22'. "
                f"Choose one of: {list(_REPLOGLE22_VARIANT_PATHS)}."
            )
        if dataset_path is not None:
            raise ValueError(
                "Do not provide dataset_path for dataset_name='replogle22'; "
                "the path is selected by dataset_variant."
            )
        if dataset_variant not in _REPLOGLE22_VARIANT_PATHS:
            raise ValueError(
                f"Unknown Replogle22 dataset_variant={dataset_variant!r}. "
                f"Choose one of: {list(_REPLOGLE22_VARIANT_PATHS)}."
            )
        return _REPLOGLE22_VARIANT_PATHS[dataset_variant], dataset_variant

    if dataset_variant is not None:
        raise ValueError("dataset_variant is only supported for dataset_name='replogle22'.")
    if dataset_path is None:
        if dataset_name == "norman19":
            return _DEFAULT_DATASET_PATH, None
        raise ValueError(f"dataset_path is required for dataset_name={dataset_name!r}.")
    return dataset_path, None


def _dataset_run_name(dataset_name: str, dataset_variant: str | None) -> str:
    """Return the dataset label used for run-scoped artifacts."""
    if dataset_variant is None:
        return dataset_name
    return f"{dataset_name}_{dataset_variant}"


def _trial_dataset_name(dataset_run_name: str, trial_id: int) -> str:
    """Return the dataset label used by one trial's model loggers."""
    return f"{dataset_run_name}_trial_{trial_id}"


def _validate_replogle22_variant(
    adata: ad.AnnData,
    dataset_variant: str,
) -> None:
    """Validate that a Replogle22 input contains K562 and its selected partner."""
    if "cell_line" not in adata.obs:
        raise KeyError("Replogle22 data must contain a 'cell_line' column in adata.obs.")

    actual_cell_lines = set(adata.obs["cell_line"].astype(str).unique())
    expected_cell_lines = {"K562", dataset_variant}
    if actual_cell_lines != expected_cell_lines:
        raise ValueError(
            f"Replogle22 variant {dataset_variant!r} must contain exactly "
            f"{sorted(expected_cell_lines)} in obs['cell_line']; found "
            f"{sorted(actual_cell_lines)}."
        )


def _concat_nonempty_adatas(adatas: list[ad.AnnData]) -> ad.AnnData:
    """Concatenate non-empty split objects while preserving shared metadata."""
    nonempty = [adata_obj for adata_obj in adatas if adata_obj.n_obs > 0]
    if not nonempty:
        raise ValueError("Expected at least one non-empty AnnData object to concatenate.")
    if len(nonempty) == 1:
        return nonempty[0]
    return ad.concat(
        nonempty,
        join="inner",
        merge="same",
        uns_merge="same",
        index_unique=None,
    )


def _release_process_memory() -> None:
    """Release unreachable objects, CUDA cache, and free Linux heap pages."""
    gc.collect()
    try:
        cuda_module = cast(_CudaModule, importlib.import_module("torch.cuda"))
    except ModuleNotFoundError:
        cuda_module = None
    if cuda_module is not None:
        if cuda_module.is_available():
            cuda_module.empty_cache()
    if sys.platform == "linux":
        ctypes.CDLL(None).malloc_trim(0)


def _context_mask(
    obs: pd.DataFrame,
    context_axis: str,
    context_values: tuple[Any, ...],
) -> np.ndarray:
    """Return a boolean mask for rows matching the requested context values."""
    return np.isin(obs[context_axis].to_numpy(copy=False), context_values)


def _filter_evaluation_bucket_min_cells(
    adata_obj: ad.AnnData,
    *,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    min_cells_per_non_control: int = 2,
    min_cells_for_control: int = 2,
) -> tuple[ad.AnnData, tuple[str, ...]]:
    """Drop evaluation perturbation groups that are too small for DE analysis."""
    labels = adata_obj.obs[perturbation_key].astype(str)
    counts = labels.value_counts(sort=False)

    control_count = int(counts.get(control_label, 0))
    if control_count < int(min_cells_for_control):
        return adata_obj[[], :].copy(), tuple()

    dropped = tuple(
        sorted(
            label
            for label, count in counts.items()
            if label != control_label and int(count) < int(min_cells_per_non_control)
        )
    )
    if not dropped:
        return adata_obj, tuple()

    keep_mask = ~labels.isin(dropped)
    return adata_obj[keep_mask.to_numpy(), :].copy(), dropped


def _filter_prediction_to_eval_perturbations(
    pred: ad.AnnData,
    perturbation_ids: np.ndarray,
    *,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
) -> ad.AnnData:
    """Restrict predictions to evaluated perturbations plus controls."""
    allowed = np.concatenate(
        [
            np.asarray([control_label], dtype=object),
            np.asarray(perturbation_ids, dtype=object),
        ]
    )
    keep_mask = np.isin(
        pred.obs[perturbation_key].to_numpy(copy=False),
        allowed,
    )
    return pred[keep_mask, :].copy()


def _copy_prediction_matrix(matrix: Any) -> Any:
    """Copy dense or sparse prediction matrices into standard in-memory form."""
    if sparse.issparse(matrix):
        return matrix.copy().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32).copy()


def _make_prediction_adata(
    *,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    prediction_matrix: Any,
    layer_name: str | None,
) -> ad.AnnData:
    """Build a lightweight prediction AnnData aligned to test metadata."""
    copied_obs = obs.copy()
    copied_var = var.copy()
    values = _copy_prediction_matrix(prediction_matrix)
    if layer_name is None:
        return ad.AnnData(X=values, obs=copied_obs, var=copied_var)

    pred = ad.AnnData(
        X=sparse.csr_matrix((copied_obs.shape[0], copied_var.shape[0]), dtype=np.float32),
        obs=copied_obs,
        var=copied_var,
    )
    pred.layers[layer_name] = values
    return pred


def _dataset_sparsity(adata_obj: ad.AnnData) -> float:
    matrix = adata_obj.X
    total_entries = int(adata_obj.n_obs) * int(adata_obj.n_vars)
    if total_entries == 0:
        return float("nan")

    if sparse.issparse(matrix):
        nonzero = int(matrix.nnz)
    else:
        nonzero = int(np.count_nonzero(np.asarray(matrix)))
    return 1.0 - (nonzero / total_entries)


def _prepare_trial_data(
    *,
    train_adata: ad.AnnData,
    val_adata: ad.AnnData,
    test_adata: ad.AnnData,
    split_metadata: Any,
    obs_layer: str | None,
) -> PreparedTrialData:
    """Prepare split-level statistics and evaluation buckets for one trial."""
    train_val_adata = _concat_nonempty_adatas([train_adata, val_adata])

    test_perturbation_ids = get_eval_perturbation_ids(
        obs=test_adata,
        control_label="control",
        strict_match=False,
    )
    if test_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbations found in test split.")

    train_perturbation_ids = get_eval_perturbation_ids(
        obs=train_val_adata,
        control_label="control",
        strict_match=False,
    )
    if train_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbations found in train/val split.")

    mu_train = compute_means_by_perturbation(
        adata_view=train_val_adata,
        perturbation_ids=train_perturbation_ids,
        layer_key=obs_layer,
        missing_group_context="evaluation view",
    )
    control_id = np.asarray(["control"], dtype=object)
    mu_control_train = compute_means_by_perturbation(
        adata_view=train_val_adata,
        perturbation_ids=control_id,
        layer_key=obs_layer,
        missing_group_context="evaluation view",
    )[0]
    train_val_labels = train_val_adata.obs["perturbation"].to_numpy(copy=False)
    train_counts = np.array(
        [np.sum(train_val_labels == pert_id) for pert_id in train_perturbation_ids],
        dtype=np.float64,
    )
    mu_pool_train = np.average(mu_train, axis=0, weights=train_counts).astype(
        np.float32, copy=False
    )
    train_targets = [label_to_target_tokens(str(pid)) for pid in train_perturbation_ids]
    test_targets = [label_to_target_tokens(str(pid)) for pid in test_perturbation_ids]
    gene_names = train_val_adata.var_names.to_numpy()

    train_labels = train_adata.obs["perturbation"].to_numpy(copy=False)
    test_labels = test_adata.obs["perturbation"].to_numpy(copy=False)
    train_perturbed_mask = train_labels != "control"

    eval_specs: list[dict[str, Any]] = []
    for eval_context in list(split_metadata.evaluation_contexts):
        bucket_test_mask = _context_mask(
            obs=test_adata.obs,
            context_axis=eval_context.axis,
            context_values=eval_context.values,
        )
        bucket_control_mask = bucket_test_mask & (np.asarray(test_labels) == "control")
        bucket_train_context_mask = (
            _context_mask(
                obs=train_adata.obs,
                context_axis=eval_context.axis,
                context_values=eval_context.values,
            )
            & train_perturbed_mask
        )

        if not np.any(bucket_test_mask) or not np.any(bucket_control_mask):
            continue
        if not np.any(bucket_train_context_mask):
            raise ValueError(
                "No non-control training cells found for context-specific baselines in "
                f"context {eval_context.value_label!r}."
            )

        bucket_obs_eval = test_adata[bucket_test_mask, :].copy()
        bucket_obs_eval, _ = _filter_evaluation_bucket_min_cells(
            bucket_obs_eval,
            perturbation_key="perturbation",
            control_label="control",
        )
        bucket_perturbation_ids = get_eval_perturbation_ids(
            obs=bucket_obs_eval,
            control_label="control",
            strict_match=False,
        )
        if bucket_perturbation_ids.size == 0:
            continue

        bucket_mu_obs = compute_means_by_perturbation(
            adata_view=bucket_obs_eval,
            perturbation_ids=bucket_perturbation_ids,
            layer_key=obs_layer,
            missing_group_context="evaluation view",
        )
        bucket_mu_control = compute_means_by_perturbation(
            adata_view=bucket_obs_eval,
            perturbation_ids=control_id,
            layer_key=obs_layer,
            missing_group_context="evaluation view",
        )[0]
        bucket_labels = bucket_obs_eval.obs["perturbation"].to_numpy(copy=False)
        bucket_has_control = bool(np.any(np.asarray(bucket_labels) == "control"))
        bucket_counts = np.array(
            [np.sum(bucket_labels == pert_id) for pert_id in bucket_perturbation_ids],
            dtype=np.float64,
        )
        bucket_mu_pool = np.average(
            bucket_mu_obs,
            axis=0,
            weights=bucket_counts,
        ).astype(np.float32, copy=False)

        bucket_train_context = train_adata[bucket_train_context_mask, :].copy()
        bucket_train_context_perturbation_ids = get_eval_perturbation_ids(
            obs=bucket_train_context,
            control_label="control",
            strict_match=False,
        )
        bucket_mu_train_context = compute_means_by_perturbation(
            adata_view=bucket_train_context,
            perturbation_ids=bucket_train_context_perturbation_ids,
            layer_key=obs_layer,
            missing_group_context="training context view",
        )
        bucket_train_context_labels = bucket_train_context.obs["perturbation"].to_numpy(copy=False)
        bucket_train_context_counts = np.array(
            [
                np.sum(bucket_train_context_labels == pert_id)
                for pert_id in bucket_train_context_perturbation_ids
            ],
            dtype=np.float64,
        )
        bucket_mu_pool_train_context = np.average(
            bucket_mu_train_context,
            axis=0,
            weights=bucket_train_context_counts,
        ).astype(np.float32, copy=False)
        bucket_train_context_targets = [
            label_to_target_tokens(str(pid)) for pid in bucket_train_context_perturbation_ids
        ]
        bucket_test_targets = [label_to_target_tokens(str(pid)) for pid in bucket_perturbation_ids]
        bucket_de_table = scanpy_de_table(
            adata=bucket_obs_eval.copy(),
            pert_col="perturbation",
            control_pert="control",
            key_added="test_de",
            layer=obs_layer,
        )
        bucket_degs = de_table_to_deg_masks(
            de_table=bucket_de_table,
            gene_names=test_adata.var_names,
            perturbation_ids=bucket_perturbation_ids,
            fdr_threshold=0.05,
        )
        bucket_mmd_pca_model = None
        if bucket_has_control:
            bucket_mmd_pca_model = fit_control_incremental_pca(
                data_obj=bucket_obs_eval,
                layer_key=obs_layer,
                control_label="control",
                obs_key="perturbation",
                data_name="bucket_obs_eval",
            )
        if bucket_mmd_pca_model is None:
            raise ValueError("Vendi calibration requires observed control cells.")
        if obs_layer is None:
            raise ValueError("Vendi calibration requires an explicit observed layer.")
        bucket_mmd_gamma = estimate_mmd_gamma(
            obs=bucket_obs_eval,
            layer_obs=obs_layer,
            control_label="control",
            pca_model=bucket_mmd_pca_model,
        )
        bucket_vendi_outer_sigma_squared = estimate_vendi_outer_sigma_squared(
            ac=bucket_obs_eval,
            gamma=bucket_mmd_gamma,
            pca_model=bucket_mmd_pca_model,
            layer_key=obs_layer,
            control_label="control",
        )
        bucket_vendi_pseudobulk_pca_model = None
        bucket_vendi_pseudobulk_sigma_squared = None
        if bucket_mu_obs.shape[0] > 1:
            bucket_vendi_pseudobulk_pca_model = fit_vendi_pseudobulk_pca(bucket_mu_obs)
            bucket_vendi_pseudobulk_sigma_squared = estimate_vendi_pseudobulk_sigma_squared(
                ac=bucket_obs_eval,
                pca_model=bucket_vendi_pseudobulk_pca_model,
                layer_key=obs_layer,
                control_label="control",
            )
        eval_specs.append(
            {
                "context_axis": eval_context.axis,
                "context_values": eval_context.value_label,
                "context_value_tuple": eval_context.values,
                "obs_eval": bucket_obs_eval,
                "perturbation_ids": bucket_perturbation_ids,
                "mu_obs": bucket_mu_obs,
                "mu_control_obs": bucket_mu_control,
                "mu_pool_obs": bucket_mu_pool,
                "mu_train_context": bucket_mu_train_context,
                "mu_pool_train_context": bucket_mu_pool_train_context,
                "train_context_targets": bucket_train_context_targets,
                "test_targets": bucket_test_targets,
                "obs_DE_table": bucket_de_table,
                "obs_DEGs": bucket_degs,
                "mmd_gamma": bucket_mmd_gamma,
                "mmd_pca_model": bucket_mmd_pca_model,
                "vendi_outer_sigma_squared": bucket_vendi_outer_sigma_squared,
                "vendi_pseudobulk_pca_model": bucket_vendi_pseudobulk_pca_model,
                "vendi_pseudobulk_sigma_squared": bucket_vendi_pseudobulk_sigma_squared,
                "mu_pred_indexer": indexer_from_labels(
                    all_labels=test_perturbation_ids,
                    selected_labels=bucket_perturbation_ids,
                ),
            }
        )

    if not eval_specs:
        raise ValueError(
            "No evaluation contexts contained both held-out controls and non-control perturbations."
        )

    return PreparedTrialData(
        split_metadata=split_metadata,
        test_obs=test_adata.obs.copy(),
        test_var=test_adata.var.copy(),
        test_perturbation_ids=test_perturbation_ids,
        mu_train=mu_train,
        mu_control_train=mu_control_train,
        mu_pool_train=mu_pool_train,
        train_targets=train_targets,
        test_targets=test_targets,
        gene_names=gene_names,
        eval_specs=eval_specs,
    )


def _evaluate_trial_model(
    *,
    prepared: PreparedTrialData,
    model: str,
    trial_id: int,
    execution_time: float,
    eval_layer_name: str | None,
    mu_pred: np.ndarray | None,
    ad_test_pred: ad.AnnData | None,
) -> list[dict[str, Any]]:
    """Evaluate one model's outputs across all prepared evaluation contexts."""
    model_rows: list[dict[str, Any]] = []
    for eval_spec in prepared.eval_specs:
        pred_for_eval = ad_test_pred
        if pred_for_eval is not None and eval_spec["context_axis"] is not None:
            # Distribution-based metrics should compare predictions and observations
            # on the same context subset.
            pred_context_mask = np.isin(
                pred_for_eval.obs[eval_spec["context_axis"]].to_numpy(copy=False),
                eval_spec["context_value_tuple"],
            )
            pred_for_eval = pred_for_eval[pred_context_mask, :].copy()
        if pred_for_eval is not None:
            pred_for_eval = _filter_prediction_to_eval_perturbations(
                pred_for_eval,
                eval_spec["perturbation_ids"],
                control_label="control",
            )

        if model == "Context-Average":
            mu_pred_for_eval = np.tile(
                eval_spec["mu_pool_train_context"],
                (eval_spec["perturbation_ids"].size, 1),
            )
        elif model == "Context-linearPCA":
            mu_pred_for_eval = predict_linear_pca_baseline(
                train_means=eval_spec["mu_train_context"],
                train_target_genes=eval_spec["train_context_targets"],
                test_target_genes=eval_spec["test_targets"],
                gene_names=prepared.gene_names,
                fallback_mean=eval_spec["mu_pool_train_context"],
                seed=trial_id,
            )
        elif mu_pred is None:
            mu_pred_for_eval = None
        else:
            mu_pred_for_eval = mu_pred[eval_spec["mu_pred_indexer"], :]

        model_metrics = evaluation(
            pred=pred_for_eval,
            obs=eval_spec["obs_eval"],
            mu_obs=eval_spec["mu_obs"],
            mu_pred=mu_pred_for_eval,
            mu_control_obs=eval_spec["mu_control_obs"],
            mu_pool_obs=eval_spec["mu_pool_obs"],
            true_DEGs=None,
            obs_DE_table=eval_spec["obs_DE_table"],
            obs_DEGs=eval_spec["obs_DEGs"],
            mmd_gamma=eval_spec["mmd_gamma"],
            mmd_pca_model=eval_spec["mmd_pca_model"],
            vendi_outer_sigma_squared=eval_spec["vendi_outer_sigma_squared"],
            vendi_pseudobulk_pca_model=eval_spec["vendi_pseudobulk_pca_model"],
            vendi_pseudobulk_sigma_squared=eval_spec["vendi_pseudobulk_sigma_squared"],
            perturbation_ids=eval_spec["perturbation_ids"],
            model=model,
            layer_name=eval_layer_name,
            control_label="control",
        )
        model_metrics.update(
            {
                "model": model,
                "trial_id": int(trial_id),
                "status": "success",
                "execution_time": execution_time,
                "context_axis": eval_spec["context_axis"],
                "context_values": eval_spec["context_values"],
            }
        )
        model_rows.append(model_metrics)
    return model_rows


def _run_trial_models(
    *,
    prepared: PreparedTrialData,
    trial_id: int,
    eval_layer_name: str | None,
    run_scvi: Callable[[], ad.AnnData],
    run_gears: Callable[[], ad.AnnData],
    run_cpa: Callable[[], ad.AnnData],
    run_state: Callable[[], ad.AnnData],
    run_scldm: Callable[[], ad.AnnData],
) -> list[dict[str, Any]]:
    """Run every model for one prepared trial using backend-specific predictors."""
    trial_results: list[dict[str, Any]] = []
    for model in MODELS:
        start_time = time.time()
        mu_pred = None
        ad_test_pred = None

        if model == "Control":
            mu_pred = np.tile(
                prepared.mu_control_train,
                (prepared.test_perturbation_ids.size, 1),
            )
        elif model == "Average":
            mu_pred = np.tile(
                prepared.mu_pool_train,
                (prepared.test_perturbation_ids.size, 1),
            )
        elif model == "Context-Average":
            mu_pred = None
        elif model == "Context-linearPCA":
            mu_pred = None
        elif model == "linearPCA":
            mu_pred = predict_linear_pca_baseline(
                train_means=prepared.mu_train,
                train_target_genes=prepared.train_targets,
                test_target_genes=prepared.test_targets,
                gene_names=prepared.gene_names,
                fallback_mean=prepared.mu_pool_train,
                seed=trial_id,
            )
        elif model == "scVI":
            ad_test_pred = run_scvi()
        elif model == "GEARS":
            ad_test_pred = run_gears()
        elif model == "CPA":
            ad_test_pred = run_cpa()
        elif model == "STATE":
            ad_test_pred = run_state()
        elif model == "scLDM":
            ad_test_pred = run_scldm()
        else:
            raise NotImplementedError(f"Model '{model}' is not implemented.")

        execution_time = time.time() - start_time
        trial_results.extend(
            _evaluate_trial_model(
                prepared=prepared,
                model=model,
                trial_id=trial_id,
                execution_time=execution_time,
                eval_layer_name=eval_layer_name,
                mu_pred=mu_pred,
                ad_test_pred=ad_test_pred,
            )
        )

        if ad_test_pred is not None:
            del ad_test_pred
        gc.collect()

    return trial_results


def _trial_model_dir(
    model_name: str,
    dataset_run_name: str,
    pid: int,
    trial_id: int,
) -> str:
    """Build the output directory for one trained real-experiment model."""
    return str(
        Path("results")
        / "real_experiments"
        / model_name
        / dataset_run_name
        / str(pid)
        / f"trial_{trial_id}"
    )


def run_one_trial(
    adata: ad.AnnData,
    splitter: ContextSplitter,
    trial_id: int,
    dataset_run_name: str,
    counts_layer: str | None,
    obs_layer: str | None,
    pid: int,
    norm_target_sum: float | None,
) -> list[dict[str, Any]]:
    """Train/evaluate every model for one split and return metric rows."""
    train_idx, val_idx, test_idx = splitter.split(seed=trial_id)
    split_metadata = splitter.get_split_metadata()
    train_adata = adata[train_idx, :]
    val_adata = adata[val_idx, :]
    test_adata = adata[test_idx, :]
    context_axis = split_metadata.context_axis
    trial_dataset_name = _trial_dataset_name(dataset_run_name, trial_id)
    prepared = _prepare_trial_data(
        train_adata=train_adata,
        val_adata=val_adata,
        test_adata=test_adata,
        split_metadata=split_metadata,
        obs_layer=obs_layer,
    )

    def run_scvi_prediction() -> ad.AnnData:
        scvi_model = ScviPerturbation(
            data=adata,
            counts_layer=counts_layer,
            perturbation_key="perturbation",
            context_key=context_axis,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=trial_id,
        )
        return scvi_model.run(
            n_latent=10,
            n_hidden=128,
            n_layers=3,
            gene_likelihood="zinb",
            dispersion="gene",
            max_epochs=800,
            batch_size=512,
            early_stopping=bool(val_idx.size > 0),
            dataloader_num_workers=0,
            normalized_target_sum=norm_target_sum,
        )

    def run_gears_prediction() -> ad.AnnData:
        gears_out = run_gears(
            train_adata=adata[train_idx, :],
            valid_adata=adata[val_idx, :],
            test_adata=adata[test_idx, :],
            is_synthetic=False,
            dataset_name=trial_dataset_name,
            model_dir=_trial_model_dir("gears", dataset_run_name, pid, trial_id),
            perturbation_column="perturbation",
            control_label="control",
            expression_layer=obs_layer,
            use_gene_ontology_graph=False,
            normalized_target_sum=norm_target_sum,
        )
        gears_pred = gears_out["preds"].detach().cpu().numpy().astype(np.float32, copy=False)
        pred = _make_prediction_adata(
            obs=prepared.test_obs,
            var=prepared.test_var,
            prediction_matrix=gears_pred,
            layer_name=obs_layer,
        )
        del gears_pred
        return pred

    def run_cpa_prediction() -> ad.AnnData:
        cpa_out = run_cpa(
            train_adata=train_adata,
            valid_adata=val_adata,
            test_adata=test_adata,
            add_controls=True,
            perturbation_column="perturbation",
            control_label="control",
            use_counts=False,
            epochs=100,
            model_dir=_trial_model_dir("CPA", dataset_run_name, pid, trial_id),
            dataset_name=trial_dataset_name,
            expression_layer=obs_layer,
            covariate_keys=[context_axis],
            library_size=None,
        )
        cpa_pred = np.asarray(cpa_out["preds"], dtype=np.float32)
        pred = _make_prediction_adata(
            obs=prepared.test_obs,
            var=prepared.test_var,
            prediction_matrix=cpa_pred,
            layer_name=obs_layer,
        )
        del cpa_pred
        return pred

    def run_state_prediction() -> ad.AnnData:
        state_out = run_state_gene(
            train_adata=train_adata,
            valid_adata=val_adata,
            test_adata=test_adata,
            context_key=context_axis,
            dataset_name=trial_dataset_name,
            model_dir=_trial_model_dir("state_gene", dataset_run_name, pid, trial_id),
            perturbation_column="perturbation",
            control_label="control",
            expression_layer=NORM_LAYER_KEY,
            epochs=100,
        )
        state_pred = np.asarray(state_out["preds"], dtype=np.float32)
        pred = _make_prediction_adata(
            obs=prepared.test_obs,
            var=prepared.test_var,
            prediction_matrix=state_pred,
            layer_name=obs_layer,
        )
        del state_pred
        return pred

    def run_scldm_prediction() -> ad.AnnData:
        scldm_out = run_scldm(
            train_adata=train_adata,
            val_adata=val_adata,
            test_adata=test_adata,
            out_dir=_trial_model_dir("scLDM", dataset_run_name, pid, trial_id),
            perturbation_column="perturbation",
            context_key=context_axis,
            control_label="control",
            num_epochs=100,
            seed=trial_id,
            resume=False,
            counts_layer=counts_layer,
            normalized_target_sum=norm_target_sum if obs_layer is not None else None,
        )
        scldm_pred = np.asarray(scldm_out["preds"], dtype=np.float32)
        pred = _make_prediction_adata(
            obs=prepared.test_obs,
            var=prepared.test_var,
            prediction_matrix=scldm_pred,
            layer_name=obs_layer,
        )
        del scldm_pred
        del scldm_out
        return pred

    return _run_trial_models(
        prepared=prepared,
        trial_id=trial_id,
        eval_layer_name=obs_layer,
        run_scvi=run_scvi_prediction,
        run_gears=run_gears_prediction,
        run_cpa=run_cpa_prediction,
        run_state=run_state_prediction,
        run_scldm=run_scldm_prediction,
    )


def _build_run_identifier(
    dataset_name: str,
    dataset_variant: str | None,
    split_strategy: str,
) -> str:
    """Build the shared timestamped identifier for results and caches."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_run_name = _dataset_run_name(dataset_name, dataset_variant)
    if dataset_name == "Norman19":
        return f"{dataset_run_name}_{timestamp}"
    return f"{dataset_run_name}_{split_strategy}_{timestamp}"


def _write_results(
    *,
    all_rows: list[dict[str, Any]],
    csv_file: Path,
    error_log_file: Path,
    n_trials: int,
) -> str:
    """Persist results rows and summarize trial-level success/failure counts."""
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(csv_file, index=False)

    if "status" in results_df.columns:
        failed_df = results_df[results_df["status"] == "failed"]
        if not failed_df.empty:
            with error_log_file.open("w", encoding="utf-8") as handle:
                for _, row in failed_df.iterrows():
                    handle.write(f"Trial {int(row['trial_id']) + 1} ({row['model']}) failed\n")
                    handle.write(f"Error: {row.get('error', 'Unknown error')}\n")
                    handle.write("-" * 80 + "\n")
            print(f"Some trials failed. Error log: {error_log_file}")

    success_trials = 0
    if "status" in results_df.columns and "model" in results_df.columns:
        success_trials = int(
            results_df[(results_df["status"] == "success") & (results_df["model"] == MODELS[0])][
                "trial_id"
            ].nunique()
        )
    failed_trials = int(n_trials) - int(success_trials)

    print(f"Done. Results saved to: {csv_file}")
    print(f"Success: {success_trials}/{n_trials} trials")
    print(f"Failed: {failed_trials}/{n_trials} trials")
    return str(csv_file)


def _execute_trial_loop(
    *,
    n_trials: int,
    splitter: ContextSplitter,
    summary: DatasetSummary,
    run_trial: Any,
) -> list[dict[str, Any]]:
    """Run one dataset's trial loop with shared success/failure bookkeeping."""
    summary_payload = {
        "dataset": summary.dataset,
        "dataset_variant": summary.dataset_variant,
        "dataset_path": summary.dataset_path,
        "n_cells": int(summary.n_cells),
        "n_genes": int(summary.n_genes),
        "n_cell_lines": int(summary.n_cell_lines),
        "n_total_perturbations": int(summary.n_total_perturbations),
        "sparsity": summary.sparsity,
    }

    all_rows: list[dict[str, Any]] = []
    for trial_id in range(int(n_trials)):
        print(f"Trial {trial_id + 1}/{n_trials}")
        try:
            trial_rows = run_trial(trial_id)
            for row in trial_rows:
                row.update(summary_payload)
            all_rows.extend(trial_rows)
        except Exception as exc:
            error_row = {
                **summary_payload,
                "trial_id": int(trial_id),
                "status": "failed",
                "error": str(exc),
                "execution_time": np.nan,
            }
            error_contexts = [
                (context.axis, context.value_label)
                for context in build_evaluation_contexts(splitter, seed=trial_id)
            ]
            for context_axis, context_values in error_contexts:
                for model in MODELS:
                    row = error_row.copy()
                    row["model"] = model
                    row["context_axis"] = context_axis
                    row["context_values"] = context_values
                    for metric_key in _METRIC_COLUMNS:
                        row[metric_key] = np.nan
                    all_rows.append(row)
        finally:
            _release_process_memory()
    return all_rows


def _resolve_real_split_strategy(
    *,
    split_strategy: str,
    context_axis: str,
    n_context_values: int,
) -> str:
    """Adjust unsupported real-data split requests before trial execution."""
    if split_strategy == "cross-context" and n_context_values == 1:
        print(
            f"Warning: 'cross-context' split strategy is not applicable when "
            f"'{context_axis}' has only one value. Defaulting to 'in-context'."
        )
        return "in-context"
    return split_strategy


def _run_real_experiments_with_runtime(
    *,
    dataset_name: str,
    dataset_variant: str | None,
    dataset_path: str,
    output_dir: str,
    n_trials: int,
    split_strategy: str,
    splitter_adata: Any,
    n_obs: int,
    n_vars: int,
    n_total_perturbations: int,
    sparsity: float,
    run_trial: Callable[[ContextSplitter, int], list[dict[str, Any]]],
    extra_log_lines: tuple[str, ...] = (),
) -> str:
    """Run the shared real-experiment loop once the dataset/runtime is prepared."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    identifier = _build_run_identifier(
        dataset_name,
        dataset_variant,
        split_strategy,
    )
    csv_file = output_dir_path / f"results_{identifier}.csv"
    error_log_file = output_dir_path / f"error_log_{identifier}.txt"

    context_config = get_dataset_context_config(dataset_name)
    context_axis = context_config.context_axis
    n_context_values = int(splitter_adata.obs[context_axis].nunique())
    split_strategy = _resolve_real_split_strategy(
        split_strategy=split_strategy,
        context_axis=context_axis,
        n_context_values=n_context_values,
    )

    dataset_run_name = _dataset_run_name(dataset_name, dataset_variant)
    print(f"Loaded dataset '{dataset_run_name}' from {dataset_path}")
    print(f"Shape: cells={n_obs}, genes={n_vars}")
    print(f"Contexts: {n_context_values} unique ({context_axis})")
    print(f"Perturbations (non-control): {n_total_perturbations}")
    for log_line in extra_log_lines:
        print(log_line)
    print(f"Running {n_trials} trials sequentially (no multiprocessing).")

    summary = DatasetSummary(
        dataset=dataset_name,
        dataset_variant=dataset_variant,
        dataset_path=dataset_path,
        n_cells=int(n_obs),
        n_genes=int(n_vars),
        n_cell_lines=n_context_values,
        n_total_perturbations=n_total_perturbations,
        sparsity=sparsity,
    )
    splitter = ContextSplitter(
        adata=splitter_adata,
        split_strategy=split_strategy,
        context_axis=context_axis,
        test_context_values=context_config.heldout_values,
    )
    all_rows = _execute_trial_loop(
        n_trials=int(n_trials),
        splitter=splitter,
        summary=summary,
        run_trial=lambda trial_id: run_trial(splitter, trial_id),
    )

    return _write_results(
        all_rows=all_rows,
        csv_file=csv_file,
        error_log_file=error_log_file,
        n_trials=int(n_trials),
    )


def _materialize_cd4_split_cache(
    *,
    runtime: CD4ChunkedDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    obs_layer: str | None,
    trial_cache_dir: Path,
    counts_layer: str | None,
) -> dict[str, Path]:
    """Write normalized split-scoped cache files for one CD4 trial."""
    if not runtime.has_source_layer(obs_layer):
        raise KeyError(
            f"Requested obs_layer='{obs_layer}' is not available in the source CD4 chunks."
        )

    split_paths = {
        "train": trial_cache_dir / "train.h5ad",
        "val": trial_cache_dir / "val.h5ad",
        "test": trial_cache_dir / "test.h5ad",
    }
    trial_cache_dir.mkdir(parents=True, exist_ok=True)

    included_layers = (counts_layer,) if counts_layer is not None else ()
    for split_name, indices in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        subset = runtime.materialize_subset(
            indices=indices,
            x_layer=obs_layer,
            include_layers=included_layers,
            output_path=split_paths[split_name],
        )
        del subset
        gc.collect()

    return split_paths


def _read_cd4_split_cache(
    split_paths: dict[str, Path],
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Load one materialized CD4 train/val/test cache triplet into memory."""
    return (
        ad.read_h5ad(split_paths["train"]),
        ad.read_h5ad(split_paths["val"]),
        ad.read_h5ad(split_paths["test"]),
    )


def _prepare_cd4_trial_data_from_cache(
    *,
    split_paths: dict[str, Path],
    split_metadata: Any,
) -> PreparedTrialData:
    """Build shared per-trial evaluation state from cached CD4 splits."""
    train_cached, val_cached, test_cached = _read_cd4_split_cache(split_paths)
    prepared = _prepare_trial_data(
        train_adata=train_cached,
        val_adata=val_cached,
        test_adata=test_cached,
        split_metadata=split_metadata,
        obs_layer=None,
    )
    del train_cached
    del val_cached
    del test_cached
    gc.collect()
    return prepared


def _run_model_from_cd4_split_cache(
    *,
    split_paths: dict[str, Path],
    run_model: Callable[[ad.AnnData, ad.AnnData, ad.AnnData], Any],
) -> Any:
    """Load one CD4 cache triplet, run a model, and release the split objects."""
    train_split, val_split, test_split = _read_cd4_split_cache(split_paths)
    model_output = run_model(train_split, val_split, test_split)
    del train_split
    del val_split
    del test_split
    gc.collect()
    return model_output


def run_one_trial_cd4_chunked(
    *,
    runtime: CD4ChunkedDataset,
    splitter: ContextSplitter,
    trial_id: int,
    dataset_run_name: str,
    counts_layer: str | None,
    obs_layer: str | None,
    pid: int,
    norm_target_sum: float | None,
    split_cache_parent: Path,
) -> list[dict[str, Any]]:
    """Train/evaluate one CD4 trial without building a full-dataset AnnData."""
    train_idx, val_idx, test_idx = splitter.split(seed=trial_id)
    split_metadata = splitter.get_split_metadata()
    context_axis = split_metadata.context_axis
    trial_dataset_name = _trial_dataset_name(dataset_run_name, trial_id)
    with TemporaryDirectory(
        prefix=f"trial_{trial_id}_",
        dir=split_cache_parent,
    ) as trial_cache_dir_str:
        split_paths = _materialize_cd4_split_cache(
            runtime=runtime,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            obs_layer=obs_layer,
            counts_layer=counts_layer,
            trial_cache_dir=Path(trial_cache_dir_str),
        )
        prepared = _prepare_cd4_trial_data_from_cache(
            split_paths=split_paths,
            split_metadata=split_metadata,
        )

        def run_scvi_prediction() -> ad.AnnData:
            with runtime.open_collection() as handle:
                scvi_model = ScviPerturbation(
                    data=handle.collection,
                    counts_layer=counts_layer,
                    perturbation_key="perturbation",
                    context_key=context_axis,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    seed=trial_id,
                )
                scvi_pred = scvi_model.run(
                    n_latent=10,
                    n_hidden=128,
                    n_layers=3,
                    gene_likelihood="zinb",
                    dispersion="gene",
                    max_epochs=800,
                    batch_size=512,
                    early_stopping=bool(val_idx.size > 0),
                    dataloader_num_workers=0,
                    normalized_target_sum=norm_target_sum,
                )
            pred = _make_prediction_adata(
                obs=prepared.test_obs,
                var=prepared.test_var,
                prediction_matrix=get_matrix(scvi_pred, NORM_LAYER_KEY),
                layer_name=None,
            )
            del scvi_pred
            return pred

        def run_gears_prediction() -> ad.AnnData:
            gears_out = _run_model_from_cd4_split_cache(
                split_paths=split_paths,
                run_model=lambda train_split, val_split, test_split: run_gears(
                    train_adata=train_split,
                    valid_adata=val_split,
                    test_adata=test_split,
                    is_synthetic=False,
                    dataset_name=trial_dataset_name,
                    model_dir=_trial_model_dir("gears", dataset_run_name, pid, trial_id),
                    perturbation_column="perturbation",
                    control_label="control",
                    expression_layer=None,
                    use_gene_ontology_graph=False,
                    normalized_target_sum=norm_target_sum,
                ),
            )
            gears_pred = gears_out["preds"].detach().cpu().numpy().astype(np.float32, copy=False)
            pred = _make_prediction_adata(
                obs=prepared.test_obs,
                var=prepared.test_var,
                prediction_matrix=gears_pred,
                layer_name=None,
            )
            del gears_pred
            return pred

        def run_cpa_prediction() -> ad.AnnData:
            cpa_out = _run_model_from_cd4_split_cache(
                split_paths=split_paths,
                run_model=lambda train_split, val_split, test_split: run_cpa(
                    train_adata=train_split,
                    valid_adata=val_split,
                    test_adata=test_split,
                    add_controls=True,
                    perturbation_column="perturbation",
                    control_label="control",
                    use_counts=False,
                    epochs=100,
                    model_dir=_trial_model_dir("CPA", dataset_run_name, pid, trial_id),
                    dataset_name=trial_dataset_name,
                    expression_layer=None,
                    covariate_keys=[context_axis],
                    library_size=None,
                ),
            )
            cpa_pred = np.asarray(cpa_out["preds"], dtype=np.float32)
            pred = _make_prediction_adata(
                obs=prepared.test_obs,
                var=prepared.test_var,
                prediction_matrix=cpa_pred,
                layer_name=None,
            )
            del cpa_pred
            return pred

        def run_state_prediction() -> ad.AnnData:
            state_out = _run_model_from_cd4_split_cache(
                split_paths=split_paths,
                run_model=lambda train_split, val_split, test_split: run_state_gene(
                    train_adata=train_split,
                    valid_adata=val_split,
                    test_adata=test_split,
                    context_key=context_axis,
                    dataset_name=trial_dataset_name,
                    model_dir=_trial_model_dir("state_gene", dataset_run_name, pid, trial_id),
                    perturbation_column="perturbation",
                    control_label="control",
                    expression_layer=None,
                    epochs=100,
                ),
            )
            state_pred = np.asarray(state_out["preds"], dtype=np.float32)
            pred = _make_prediction_adata(
                obs=prepared.test_obs,
                var=prepared.test_var,
                prediction_matrix=state_pred,
                layer_name=None,
            )
            del state_pred
            return pred

        def run_scldm_prediction() -> ad.AnnData:
            if counts_layer is None:
                raise ValueError("scLDM requires a raw-count layer for chunked CD4+ data.")
            scldm_out = _run_model_from_cd4_split_cache(
                split_paths=split_paths,
                run_model=lambda train_split, val_split, test_split: run_scldm(
                    train_adata=train_split,
                    val_adata=val_split,
                    test_adata=test_split,
                    out_dir=_trial_model_dir("scLDM", dataset_run_name, pid, trial_id),
                    perturbation_column="perturbation",
                    context_key=context_axis,
                    control_label="control",
                    num_epochs=100,
                    seed=trial_id,
                    resume=False,
                    counts_layer=counts_layer,
                    normalized_target_sum=(norm_target_sum if obs_layer is not None else None),
                ),
            )
            scldm_pred = np.asarray(scldm_out["preds"], dtype=np.float32)
            pred = _make_prediction_adata(
                obs=prepared.test_obs,
                var=prepared.test_var,
                prediction_matrix=scldm_pred,
                layer_name=None,
            )
            del scldm_pred
            del scldm_out
            return pred

        return _run_trial_models(
            prepared=prepared,
            trial_id=trial_id,
            eval_layer_name=None,
            run_scvi=run_scvi_prediction,
            run_gears=run_gears_prediction,
            run_cpa=run_cpa_prediction,
            run_state=run_state_prediction,
            run_scldm=run_scldm_prediction,
        )


def _run_real_experiments_h5ad(
    dataset_name: str,
    dataset_variant: str | None,
    dataset_path: str,
    output_dir: str,
    n_trials: int,
    counts_layer: str | None,
    obs_layer: str | None,
    split_strategy: str,
    norm_target_sum: float,
) -> str:
    """Run the existing in-memory AnnData workflow for standard .h5ad datasets."""
    pid = os.getpid()
    adata, label_mapping = load_real_dataset(dataset_path=dataset_path)
    if dataset_variant is not None:
        _validate_replogle22_variant(adata, dataset_variant)
    dataset_run_name = _dataset_run_name(dataset_name, dataset_variant)

    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(
            f"Requested counts_layer='{counts_layer}' not found. Available layers: {list(adata.layers.keys())}"
        )

    if obs_layer is not None and obs_layer not in adata.layers:
        if obs_layer == NORM_LAYER_KEY:
            source_layer = counts_layer
            if source_layer is None and "counts" in adata.layers:
                source_layer = "counts"
            ensure_normalized_log1p_layer(
                adata=adata,
                output_layer_key=obs_layer,
                source_layer=source_layer,
                target_sum=norm_target_sum,
            )
        else:
            raise KeyError(
                f"Requested obs_layer='{obs_layer}' not found. Available layers: {list(adata.layers.keys())}"
            )

    validate_perturbation_targets_subset_from_obs(
        obs=adata.obs,
        gene_names=adata.var_names,
        control_label="control",
    )

    return _run_real_experiments_with_runtime(
        dataset_name=dataset_name,
        dataset_variant=dataset_variant,
        dataset_path=dataset_path,
        output_dir=output_dir,
        n_trials=n_trials,
        split_strategy=split_strategy,
        splitter_adata=adata,
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        n_total_perturbations=len(label_mapping),
        sparsity=_dataset_sparsity(adata),
        run_trial=lambda splitter, trial_id: run_one_trial(
            adata=adata,
            splitter=splitter,
            trial_id=trial_id,
            dataset_run_name=dataset_run_name,
            counts_layer=counts_layer,
            obs_layer=obs_layer,
            pid=pid,
            norm_target_sum=norm_target_sum,
        ),
    )


def _run_real_experiments_cd4_chunked(
    dataset_name: str,
    dataset_variant: str | None,
    dataset_path: str,
    output_dir: str,
    n_trials: int,
    counts_layer: str | None,
    obs_layer: str | None,
    split_strategy: str,
    norm_target_sum: float,
) -> str:
    """Run the chunk-aware CD4 workflow from a manifest JSON."""
    pid = os.getpid()
    dataset_run_name = _dataset_run_name(dataset_name, dataset_variant)
    split_cache_parent = Path(output_dir) / _CD4_SPLIT_CACHE_DIRNAME
    split_cache_parent.mkdir(parents=True, exist_ok=True)

    runtime = CD4ChunkedDataset.from_manifest(dataset_path)
    if counts_layer is not None and not runtime.has_source_layer(counts_layer):
        raise KeyError(
            f"Requested counts_layer='{counts_layer}' not found in CD4 chunks. "
            f"Available layers: {list(runtime.available_layers)}"
        )
    if obs_layer is not None and not runtime.has_source_layer(obs_layer):
        raise KeyError(
            f"Requested obs_layer='{obs_layer}' not found in CD4 chunks. "
            f"Available layers: {list(runtime.available_layers)} and X='{NORM_LAYER_KEY}'"
        )

    label_mapping = build_perturbation_label_mapping(runtime.obs["perturbation"])
    validate_perturbation_targets_subset_from_obs(
        obs=runtime.obs,
        gene_names=runtime.var_names,
        control_label="control",
    )

    return _run_real_experiments_with_runtime(
        dataset_name=dataset_name,
        dataset_variant=dataset_variant,
        dataset_path=dataset_path,
        output_dir=output_dir,
        n_trials=n_trials,
        split_strategy=split_strategy,
        splitter_adata=runtime,
        n_obs=int(runtime.n_obs),
        n_vars=int(runtime.n_vars),
        n_total_perturbations=len(label_mapping),
        sparsity=runtime.dataset_sparsity(obs_layer),
        run_trial=lambda splitter, trial_id: run_one_trial_cd4_chunked(
            runtime=runtime,
            splitter=splitter,
            trial_id=trial_id,
            dataset_run_name=dataset_run_name,
            counts_layer=counts_layer,
            obs_layer=obs_layer,
            pid=pid,
            norm_target_sum=norm_target_sum,
            split_cache_parent=split_cache_parent,
        ),
        extra_log_lines=(
            f"CD4 trial split caches are temporary directories under: {split_cache_parent}",
        ),
    )


def run_real_experiments(
    dataset_name: str,
    dataset_path: str | None,
    output_dir: str,
    n_trials: int,
    counts_layer: str | None,
    obs_layer: str | None,
    split_strategy: str,
    norm_target_sum: float,
    dataset_variant: str | None = None,
) -> str:
    """Run all trials for one real dataset and write results/log files."""
    dataset_path, dataset_variant = _resolve_dataset_request(
        dataset_name=dataset_name,
        dataset_variant=dataset_variant,
        dataset_path=dataset_path,
    )
    dataset_suffix = Path(dataset_path).suffix.lower()
    if dataset_suffix == ".json":
        if dataset_name != "CD4+":
            raise ValueError("Chunked JSON manifests are only supported for dataset_name='CD4+'.")
        return _run_real_experiments_cd4_chunked(
            dataset_name=dataset_name,
            dataset_variant=dataset_variant,
            dataset_path=dataset_path,
            output_dir=output_dir,
            n_trials=n_trials,
            counts_layer=counts_layer,
            obs_layer=obs_layer,
            split_strategy=split_strategy,
            norm_target_sum=norm_target_sum,
        )
    return _run_real_experiments_h5ad(
        dataset_name=dataset_name,
        dataset_variant=dataset_variant,
        dataset_path=dataset_path,
        output_dir=output_dir,
        n_trials=n_trials,
        counts_layer=counts_layer,
        obs_layer=obs_layer,
        split_strategy=split_strategy,
        norm_target_sum=norm_target_sum,
    )


def real_exp_args(description: str) -> argparse.ArgumentParser:
    """
    Build a parser for real-experiment dataset and split configuration.

    Args:
        description: CLI description to show in ``--help`` output.

    Returns:
        A configured argument parser for real-experiment scripts.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset_name", type=str, default="norman19")
    parser.add_argument(
        "--dataset_variant",
        type=str,
        choices=list(_REPLOGLE22_VARIANT_PATHS),
        help="Replogle22 subset to run. Required when dataset_name is 'replogle22'.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Input dataset path. Omit for Replogle22 and the default Norman19 dataset.",
    )
    parser.add_argument("--n_trials", type=int, default=10)

    parser.add_argument(
        "--counts_layer",
        type=str,
        default="counts",
        help="Layer containing raw counts for scVI and scLDM. Set to 'none' to use adata.X as counts.",
    )
    parser.add_argument(
        "--obs_layer",
        type=str,
        default=NORM_LAYER_KEY,
        help="Layer to use for evaluation and modeling.",
    )
    parser.add_argument(
        "--norm_target_sum",
        type=float,
        default=1e4,
        help="Target sum for normalization when obs_layer is missing and needs to be computed from counts.",
    )

    parser.add_argument(
        "--split_strategy",
        type=str,
        default="in-context",
        choices=["in-context", "cross-context"],
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line options for running real experiments."""
    return real_exp_args(
        description="Run real-world perturbation experiments with perturbation-level split by cell type."
    ).parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    counts_layer = None if str(args.counts_layer).lower() == "none" else args.counts_layer
    obs_layer = None if str(args.obs_layer).lower() == "none" else args.obs_layer

    run_real_experiments(
        dataset_name=args.dataset_name,
        dataset_variant=args.dataset_variant,
        dataset_path=args.dataset_path,
        output_dir=_DEFAULT_OUTPUT_DIR,
        n_trials=int(args.n_trials),
        counts_layer=counts_layer,
        obs_layer=obs_layer,
        split_strategy=args.split_strategy,
        norm_target_sum=float(args.norm_target_sum),
    )


if __name__ == "__main__":
    main()
