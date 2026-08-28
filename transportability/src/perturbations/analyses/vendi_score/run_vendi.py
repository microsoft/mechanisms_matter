"""Compute Vendi and split-half PDS scores for real and synthetic datasets."""

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ...data.cd4_chunked import CD4ChunkedDataset
from ...data.dgp import causalDGP
from ...metrics.perturbation_effect.perturbation_discrimination_score import pds
from ...metrics.reconstruction.distance_util import estimate_mmd_gamma
from ...metrics.reconstruction.vendi_score import (
    estimate_vendi_outer_sigma_squared,
    estimate_vendi_pseudobulk_sigma_squared,
    fit_vendi_pseudobulk_pca,
    vendi_score,
    vendi_score_pseudobulk,
)
from ...util.anndata_util import extract_rows, fit_control_incremental_pca
from ..common import NORM_LAYER_KEY
from ..context import get_dataset_context_config
from ..synthetic_simulations.sampling import load_parameter_estimation_inputs
from ..util import (
    count_non_control_perturbations,
    ensure_normalized_log1p_layer,
    load_real_dataset,
    validate_perturbation_targets_subset_from_obs,
)

_DEFAULT_OUTPUT_DIR = "results/vendi_score"
_CONTROL_LABEL = "control"
_CD4_DATASET = "CD4+"
_DIVERSITY_TYPES = ("A", "b", "both", "none")
_SYNTHETIC_CONTEXT_AXIS = "cell_line"
_REPLOGLE22_VARIANT_PATHS = {
    "RPE1": "data/replogle22/RPE1/processed.h5ad",
    "Jurkat": "data/replogle22/Jurkat/processed.h5ad",
    "HepG2": "data/replogle22/HepG2/processed.h5ad",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved input metadata for one real dataset."""

    dataset: str
    dataset_variant: str | None
    dataset_label: str
    dataset_path: str
    is_cd4_chunked: bool = False


_DATASET_SPECS = (
    DatasetSpec(
        dataset="norman19",
        dataset_variant=None,
        dataset_label="norman19",
        dataset_path="data/norman19/norman19_processed.h5ad",
    ),
    DatasetSpec(
        dataset=_CD4_DATASET,
        dataset_variant=None,
        dataset_label=_CD4_DATASET,
        dataset_path="data/cd4+/processed/processed_manifest.json",
        is_cd4_chunked=True,
    ),
    *(
        DatasetSpec(
            dataset="replogle22",
            dataset_variant=variant,
            dataset_label=f"replogle22_{variant}",
            dataset_path=path,
        )
        for variant, path in _REPLOGLE22_VARIANT_PATHS.items()
    ),
)
_DATASET_SPEC_BY_LABEL = {spec.dataset_label: spec for spec in _DATASET_SPECS}

_OUTPUT_COLUMNS = [
    "dataset",
    "dataset_variant",
    "dataset_label",
    "dataset_path",
    "scope",
    "vendi_score_cell",
    "vendi_score_pseudobulk",
    "pds_l1",
    "pds_l2",
    "pds_cosine",
    "n_cells",
    "n_genes",
    "n_total_perturbations",
    "n_contexts",
    "context_axis",
    "context_values",
    "layer_key",
    "ac_batch_size",
    "n_pca_components",
    "sample_size",
    "random_state",
    "control_label",
    "execution_time_seconds",
]


def _require_existing_path(path: str) -> None:
    """Raise a clear error when an expected input artifact is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset input does not exist: {path}")


def _estimate_vendi_params(
    ac: ad.AnnData,
    layer_key: str | None,
    control_label: str,
    n_pca_components: int,
    random_state: int,
) -> tuple[Any, float, float]:
    """Estimate shared PCA, gamma, and outer_sigma_squared from control cells."""
    pca_model = fit_control_incremental_pca(
        data_obj=ac,
        layer_key=layer_key,
        control_label=control_label,
        n_pca_components=n_pca_components,
        obs_key="perturbation",
        data_name="vendi_input",
    )
    gamma = estimate_mmd_gamma(
        obs=ac,
        layer_obs=layer_key,
        control_label=control_label,
        seed=random_state,
        pca_model=pca_model,
    )
    outer_sigma_squared = estimate_vendi_outer_sigma_squared(
        ac=ac,
        gamma=gamma,
        pca_model=pca_model,
        layer_key=layer_key,
        control_label=control_label,
        random_state=random_state,
    )
    return pca_model, gamma, outer_sigma_squared


def _pseudobulk_vendi(
    adata: ad.AnnData,
    layer_key: str | None,
    control_label: str,
    n_pca_components: int,
    random_state: int,
) -> float:
    """Compute pseudobulk-level Vendi score for one AnnData slice."""
    from ..util import compute_means_by_perturbation

    pert_labels = np.asarray(adata.obs["perturbation"])
    non_control_ids = np.unique(pert_labels[pert_labels != control_label])
    if non_control_ids.size < 2:
        return float("nan")
    mu = compute_means_by_perturbation(
        adata_view=adata,
        perturbation_ids=non_control_ids,
        layer_key=layer_key,
    )
    pb_pca = fit_vendi_pseudobulk_pca(
        mu, n_pca_components=n_pca_components, random_state=random_state
    )
    pb_sigma = estimate_vendi_pseudobulk_sigma_squared(
        ac=adata,
        pca_model=pb_pca,
        layer_key=layer_key,
        control_label=control_label,
        random_state=random_state,
    )
    return vendi_score_pseudobulk(mu, pca_model=pb_pca, outer_sigma_squared=pb_sigma)


def _split_half_pds(
    adata: ad.AnnData,
    layer_key: str | None,
    control_label: str,
    random_state: int,
) -> dict[str, float]:
    """Compute split-half PDS scores from one AnnData slice."""
    rng = np.random.default_rng(random_state)
    obs = adata.obs
    pert_labels = np.asarray(obs["perturbation"])
    unique_perts = np.unique(pert_labels)
    non_control = unique_perts[unique_perts != control_label]
    if non_control.size < 2:
        return {"pds_l1": np.nan, "pds_l2": np.nan, "pds_cosine": np.nan}

    # Control pseudobulk. `extract_rows` densifies sparse layers and reads one group at a
    # time, so a dense copy of the whole slice is never materialized.
    ctrl_mask = pert_labels == control_label
    if ctrl_mask.sum() == 0:
        return {"pds_l1": np.nan, "pds_l2": np.nan, "pds_cosine": np.nan}
    mu_control = (
        extract_rows(adata, np.flatnonzero(ctrl_mask), layer_key).mean(axis=0).reshape(1, -1)
    )

    mu_a_list: list[np.ndarray] = []
    mu_b_list: list[np.ndarray] = []
    for p in sorted(non_control.tolist(), key=str):
        idx = np.flatnonzero(pert_labels == p)
        if idx.size < 2:
            continue
        shuffled = rng.permutation(idx)
        mid = len(shuffled) // 2
        # Sorted row indices keep backed/collection slicing valid; the mean is order-invariant.
        mu_a_list.append(extract_rows(adata, np.sort(shuffled[:mid]), layer_key).mean(axis=0))
        mu_b_list.append(extract_rows(adata, np.sort(shuffled[mid:]), layer_key).mean(axis=0))

    if len(mu_a_list) < 2:
        return {"pds_l1": np.nan, "pds_l2": np.nan, "pds_cosine": np.nan}

    mu_a = np.stack(mu_a_list)
    mu_b = np.stack(mu_b_list)
    # Average both directions so the score is symmetric across splits.
    scores: dict[str, float] = {}
    for m in ("l1", "l2", "cosine"):
        fwd = pds(X_obs=mu_a, X_pred=mu_b, reference=mu_control, metric=m)
        rev = pds(X_obs=mu_b, X_pred=mu_a, reference=mu_control, metric=m)
        scores[f"pds_{m}"] = 0.5 * (fwd + rev)
    return scores


def _parse_optional_layer(layer_name: str | None) -> str | None:
    """Convert the CLI sentinel for `.X` into ``None``."""
    if layer_name is None:
        return None
    if str(layer_name).lower() == "none":
        return None
    return str(layer_name)


def _resolve_dataset_specs(dataset_labels: Sequence[str] | None) -> tuple[DatasetSpec, ...]:
    """Return the dataset specs selected by CLI label."""
    if dataset_labels is None:
        return _DATASET_SPECS

    unknown_labels = sorted(set(dataset_labels) - set(_DATASET_SPEC_BY_LABEL))
    if unknown_labels:
        raise ValueError(
            f"Unknown dataset_label(s): {unknown_labels}. "
            f"Choose from: {list(_DATASET_SPEC_BY_LABEL)}."
        )
    return tuple(_DATASET_SPEC_BY_LABEL[label] for label in dict.fromkeys(dataset_labels))


def _sorted_unique_strings(values: Any) -> tuple[str, ...]:
    """Return stable string labels for unique context values."""
    unique_values = pd.Index(pd.Series(values).astype(str).unique())
    return tuple(sorted(unique_values.tolist()))


def _context_metadata(obs: pd.DataFrame, dataset_name: str) -> tuple[str, tuple[str, ...]]:
    """Return the context axis and all observed context values for a dataset."""
    context_axis = get_dataset_context_config(dataset_name).context_axis
    if context_axis not in obs.columns:
        raise KeyError(
            f"Dataset {dataset_name!r} is missing required context column {context_axis!r}."
        )
    return context_axis, _sorted_unique_strings(obs[context_axis])


def _validate_replogle22_variant(adata: ad.AnnData, dataset_variant: str) -> None:
    """Validate that a Replogle22 input contains K562 and its selected partner."""
    if "cell_line" not in adata.obs.columns:
        raise KeyError("Replogle22 data must contain a 'cell_line' column in adata.obs.")

    actual_cell_lines = set(adata.obs["cell_line"].astype(str).unique())
    expected_cell_lines = {"K562", dataset_variant}
    if actual_cell_lines != expected_cell_lines:
        raise ValueError(
            f"Replogle22 variant {dataset_variant!r} must contain exactly "
            f"{sorted(expected_cell_lines)} in obs['cell_line']; found {sorted(actual_cell_lines)}."
        )


def _ensure_h5ad_layer(
    adata: ad.AnnData,
    obs_layer: str | None,
    counts_layer: str | None,
    norm_target_sum: float,
) -> str | None:
    """Ensure a requested h5ad expression layer exists and return it for Vendi."""
    if obs_layer is None or obs_layer in adata.layers:
        return obs_layer

    if obs_layer != NORM_LAYER_KEY:
        raise KeyError(
            f"Requested obs_layer={obs_layer!r} not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    source_layer = counts_layer
    if source_layer is None and "counts" in adata.layers:
        source_layer = "counts"
    ensure_normalized_log1p_layer(
        adata=adata,
        output_layer_key=obs_layer,
        source_layer=source_layer,
        target_sum=norm_target_sum,
    )
    return obs_layer


def _cd4_vendi_layer(runtime: CD4ChunkedDataset, obs_layer: str | None) -> str | None:
    """Resolve the layer key passed to Vendi for backed CD4 chunks."""
    if not runtime.has_source_layer(obs_layer):
        raise KeyError(
            f"Requested obs_layer={obs_layer!r} not found in CD4 chunks. "
            f"Available layers: {list(runtime.available_layers)} and X={NORM_LAYER_KEY!r}"
        )
    if obs_layer == NORM_LAYER_KEY:
        return None
    return obs_layer


def _base_result_row(
    *,
    spec: DatasetSpec,
    obs: pd.DataFrame,
    n_vars: int,
    reported_layer_key: str,
    batch_size: int,
    n_pca_components: int,
    sample_size: int,
    random_state: int,
    execution_time_seconds: float,
    score: float,
) -> dict[str, Any]:
    """Build the shared output row for one dataset."""
    context_axis, context_values = _context_metadata(obs, spec.dataset)
    return {
        "dataset": spec.dataset,
        "dataset_variant": spec.dataset_variant,
        "dataset_label": spec.dataset_label,
        "dataset_path": spec.dataset_path,
        "scope": "all",
        "vendi_score_cell": score,
        "n_cells": int(obs.shape[0]),
        "n_genes": int(n_vars),
        "n_total_perturbations": count_non_control_perturbations(
            obs,
            control_label=_CONTROL_LABEL,
        ),
        "n_contexts": len(context_values),
        "context_axis": context_axis,
        "context_values": ";".join(context_values),
        "layer_key": reported_layer_key,
        "ac_batch_size": int(batch_size),
        "n_pca_components": int(n_pca_components),
        "sample_size": int(sample_size),
        "random_state": int(random_state),
        "control_label": _CONTROL_LABEL,
        "execution_time_seconds": execution_time_seconds,
    }


def _compute_h5ad_vendi_score(
    *,
    spec: DatasetSpec,
    counts_layer: str | None,
    batch_size: int,
    n_pca_components: int,
    sample_size: int,
    random_state: int,
    norm_target_sum: float,
    by_context: bool = True,
) -> list[dict[str, Any]]:
    """Load one in-memory h5ad dataset and compute Vendi score(s)."""
    _require_existing_path(spec.dataset_path)

    adata, _ = load_real_dataset(dataset_path=spec.dataset_path)
    try:
        if spec.dataset_variant is not None:
            _validate_replogle22_variant(adata, spec.dataset_variant)

        vendi_layer_key = _ensure_h5ad_layer(
            adata=adata,
            obs_layer=NORM_LAYER_KEY,
            counts_layer=counts_layer,
            norm_target_sum=norm_target_sum,
        )
        validate_perturbation_targets_subset_from_obs(
            obs=adata.obs,
            gene_names=adata.var_names,
            control_label=_CONTROL_LABEL,
        )

        slices: list[tuple[str, ad.AnnData]] = []
        if by_context:
            context_axis, context_values = _context_metadata(adata.obs, spec.dataset)
            for ctx_val in context_values:
                mask = adata.obs[context_axis].astype(str) == ctx_val
                slices.append((ctx_val, adata[mask]))
        else:
            slices.append(("all", adata))

        rows: list[dict[str, Any]] = []
        for scope_label, adata_slice in slices:
            pca_model, gamma, outer_sigma_squared = _estimate_vendi_params(
                adata_slice, vendi_layer_key, _CONTROL_LABEL, n_pca_components, random_state
            )
            start_time = time.perf_counter()
            score = vendi_score(
                ac=adata_slice,
                ac_batch_size=batch_size,
                n_pca_components=n_pca_components,
                sample_size=sample_size,
                random_state=random_state,
                layer_key=vendi_layer_key,
                control_label=_CONTROL_LABEL,
                gamma=gamma,
                pca_model=pca_model,
                outer_sigma_squared=outer_sigma_squared,
            )
            pds_scores = _split_half_pds(adata_slice, vendi_layer_key, _CONTROL_LABEL, random_state)
            pb_vendi = _pseudobulk_vendi(
                adata_slice, vendi_layer_key, _CONTROL_LABEL, n_pca_components, random_state
            )
            execution_time_seconds = time.perf_counter() - start_time

            row = _base_result_row(
                spec=spec,
                obs=adata_slice.obs,
                n_vars=adata_slice.n_vars,
                reported_layer_key=NORM_LAYER_KEY,
                batch_size=batch_size,
                n_pca_components=n_pca_components,
                sample_size=sample_size,
                random_state=random_state,
                execution_time_seconds=execution_time_seconds,
                score=score,
            )
            row["scope"] = scope_label
            row["vendi_score_pseudobulk"] = pb_vendi
            row.update(pds_scores)
            rows.append(row)
        return rows
    finally:
        del adata
        gc.collect()


def _compute_cd4_vendi_score(
    *,
    spec: DatasetSpec,
    batch_size: int,
    n_pca_components: int,
    sample_size: int,
    random_state: int,
    by_context: bool = True,
) -> list[dict[str, Any]]:
    """Compute Vendi score(s) for CD4+ using a backed AnnCollection."""
    _require_existing_path(spec.dataset_path)

    runtime = CD4ChunkedDataset.from_manifest(spec.dataset_path)
    vendi_layer_key = _cd4_vendi_layer(runtime, NORM_LAYER_KEY)
    validate_perturbation_targets_subset_from_obs(
        obs=runtime.obs,
        gene_names=runtime.var_names,
        control_label=_CONTROL_LABEL,
    )

    context_axis, context_values = _context_metadata(runtime.obs, spec.dataset)
    if by_context:
        scopes = [(ctx_val, ctx_val) for ctx_val in context_values]
    else:
        scopes = [("all", None)]

    rows: list[dict[str, Any]] = []
    with runtime.open_collection() as handle:
        for scope_label, ctx_val in scopes:
            if ctx_val is not None:
                ctx_mask = runtime.obs[context_axis].astype(str) == ctx_val
                indices = ctx_mask.to_numpy().nonzero()[0]
                ac_slice = handle.collection[indices]
            else:
                ac_slice = handle.collection

            pca_model, gamma, outer_sigma_squared = _estimate_vendi_params(
                ac_slice, vendi_layer_key, _CONTROL_LABEL, n_pca_components, random_state
            )
            start_time = time.perf_counter()
            score = vendi_score(
                ac=ac_slice,
                ac_batch_size=batch_size,
                n_pca_components=n_pca_components,
                sample_size=sample_size,
                random_state=random_state,
                layer_key=vendi_layer_key,
                control_label=_CONTROL_LABEL,
                gamma=gamma,
                pca_model=pca_model,
                outer_sigma_squared=outer_sigma_squared,
            )
            # CD4 backed slices: materialize for PDS pseudobulk computation
            if hasattr(ac_slice, "to_adata"):
                pds_adata = ac_slice.to_adata()
            else:
                pds_adata = ac_slice
            pds_scores = _split_half_pds(pds_adata, vendi_layer_key, _CONTROL_LABEL, random_state)
            pb_vendi = _pseudobulk_vendi(
                pds_adata, vendi_layer_key, _CONTROL_LABEL, n_pca_components, random_state
            )
            execution_time_seconds = time.perf_counter() - start_time

            obs_slice = runtime.obs if ctx_val is None else runtime.obs[ctx_mask]
            row = _base_result_row(
                spec=spec,
                obs=obs_slice,
                n_vars=runtime.n_vars,
                reported_layer_key=NORM_LAYER_KEY,
                batch_size=batch_size,
                n_pca_components=n_pca_components,
                sample_size=sample_size,
                random_state=random_state,
                execution_time_seconds=execution_time_seconds,
                score=score,
            )
            row["scope"] = scope_label
            row["vendi_score_pseudobulk"] = pb_vendi
            row.update(pds_scores)
            rows.append(row)

    try:
        return rows
    finally:
        del runtime
        gc.collect()


def run_real_dataset_vendi_scores(
    *,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    dataset_labels: Sequence[str] | None = None,
    counts_layer: str | None = "counts",
    batch_size: int = 1024,
    n_pca_components: int = 50,
    sample_size: int = 2000,
    random_state: int = 0,
    norm_target_sum: float = 1e4,
    by_context: bool = True,
) -> str:
    """
    Compute whole-dataset Vendi scores for all configured real datasets.

    Args:
        output_dir: Directory where the result CSV should be written.
        dataset_labels: Dataset labels to evaluate, or ``None`` for all datasets.
        counts_layer: Count layer used only when building ``normalized_log1p`` for h5ad inputs.
        batch_size: Batch size passed to the Vendi scorer.
        n_pca_components: Number of PCA components used by the Vendi scorer.
        sample_size: Number of cells sampled for MMD bandwidth estimation.
        random_state: Random seed for deterministic sampling.
        norm_target_sum: Library-size target sum when constructing ``normalized_log1p``.
        by_context: When True, score each context value separately instead of the whole dataset.

    Returns:
        Path to the written result CSV.
    """
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive. Got {batch_size}.")
    if int(n_pca_components) <= 0:
        raise ValueError(f"n_pca_components must be positive. Got {n_pca_components}.")
    if int(sample_size) <= 0:
        raise ValueError(f"sample_size must be positive. Got {sample_size}.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / (
        f"real_dataset_vendi_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    selected_specs = _resolve_dataset_specs(dataset_labels)

    rows: list[dict[str, Any]] = []
    for spec in selected_specs:
        print(f"Computing Vendi score for {spec.dataset_label} from {spec.dataset_path}")
        if spec.is_cd4_chunked:
            spec_rows = _compute_cd4_vendi_score(
                spec=spec,
                batch_size=int(batch_size),
                n_pca_components=int(n_pca_components),
                sample_size=int(sample_size),
                random_state=int(random_state),
                by_context=by_context,
            )
        else:
            spec_rows = _compute_h5ad_vendi_score(
                spec=spec,
                counts_layer=counts_layer,
                batch_size=int(batch_size),
                n_pca_components=int(n_pca_components),
                sample_size=int(sample_size),
                random_state=int(random_state),
                norm_target_sum=float(norm_target_sum),
                by_context=by_context,
            )
        for row in spec_rows:
            rows.append(row)
            print(
                f"  [{row.get('scope', 'all')}] vendi_cell={row['vendi_score_cell']:.6g}, "
                f"cells={row['n_cells']}, genes={row['n_genes']}, "
                f"perturbations={row['n_total_perturbations']}"
            )
        pd.DataFrame(rows, columns=_OUTPUT_COLUMNS).to_csv(output_path, index=False)

    print(f"Done. Vendi scores saved to: {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Synthetic CausalDGP scoring
# ---------------------------------------------------------------------------


def _compute_synthetic_vendi_score(
    *,
    diversity_type: str,
    n_genes: int,
    n_control: int,
    n_per_perturbation: int,
    n_perturbations: int,
    batch_size: int,
    n_pca_components: int,
    sample_size: int,
    random_state: int,
    by_context: bool,
) -> list[dict[str, Any]]:
    """Generate one CausalDGP dataset and compute Vendi + PDS scores."""
    inputs = load_parameter_estimation_inputs()
    adata, _ = causalDGP(
        G=n_genes,
        N0=n_control,
        Nk=n_per_perturbation,
        P=n_perturbations,
        mu_l=1.0,
        all_theta=inputs["all_theta"],
        gene_names=inputs["gene_names"],
        mask_method="Erdos-Renyi",
        diversity_type=diversity_type,
        swap_fraction=0.5,
        seed=random_state,
        normalize=True,
        normalized_layer_key=NORM_LAYER_KEY,
    )

    # causalDGP stores raw counts in .X; always score on the normalized layer.
    layer_key: str = NORM_LAYER_KEY
    if layer_key not in adata.layers:
        raise KeyError(
            f"CausalDGP dataset is missing the '{NORM_LAYER_KEY}' layer. "
            f"Available layers: {sorted(adata.layers)}."
        )

    slices: list[tuple[str, ad.AnnData]] = []
    if by_context and _SYNTHETIC_CONTEXT_AXIS in adata.obs.columns:
        for ctx_val in sorted(adata.obs[_SYNTHETIC_CONTEXT_AXIS].astype(str).unique()):
            mask = adata.obs[_SYNTHETIC_CONTEXT_AXIS].astype(str) == ctx_val
            slices.append((ctx_val, adata[mask]))
    else:
        slices.append(("all", adata))

    rows: list[dict[str, Any]] = []
    for scope_label, adata_slice in slices:
        pca_model, gamma, outer_sigma_squared = _estimate_vendi_params(
            adata_slice, layer_key, _CONTROL_LABEL, n_pca_components, random_state
        )
        start_time = time.perf_counter()
        score = vendi_score(
            ac=adata_slice,
            ac_batch_size=batch_size,
            n_pca_components=n_pca_components,
            sample_size=sample_size,
            random_state=random_state,
            layer_key=layer_key,
            control_label=_CONTROL_LABEL,
            gamma=gamma,
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )
        pds_scores = _split_half_pds(adata_slice, layer_key, _CONTROL_LABEL, random_state)
        pb_vendi = _pseudobulk_vendi(
            adata_slice, layer_key, _CONTROL_LABEL, n_pca_components, random_state
        )
        execution_time_seconds = time.perf_counter() - start_time

        pert_labels = np.asarray(adata_slice.obs["perturbation"])
        n_perts = int(np.unique(pert_labels[pert_labels != _CONTROL_LABEL]).size)
        ctx_values = (
            sorted(adata_slice.obs[_SYNTHETIC_CONTEXT_AXIS].astype(str).unique())
            if _SYNTHETIC_CONTEXT_AXIS in adata_slice.obs.columns
            else []
        )

        row: dict[str, Any] = {
            "dataset": "causalDGP",
            "dataset_variant": diversity_type,
            "dataset_label": f"causalDGP_{diversity_type}",
            "dataset_path": "generated",
            "scope": scope_label,
            "vendi_score_cell": score,
            "vendi_score_pseudobulk": pb_vendi,
            "n_cells": int(adata_slice.n_obs),
            "n_genes": int(adata_slice.n_vars),
            "n_total_perturbations": n_perts,
            "n_contexts": len(ctx_values),
            "context_axis": _SYNTHETIC_CONTEXT_AXIS,
            "context_values": ";".join(ctx_values),
            "layer_key": layer_key,
            "ac_batch_size": batch_size,
            "n_pca_components": n_pca_components,
            "sample_size": sample_size,
            "random_state": random_state,
            "control_label": _CONTROL_LABEL,
            "execution_time_seconds": execution_time_seconds,
        }
        row.update(pds_scores)
        rows.append(row)

    del adata
    gc.collect()
    return rows


def run_synthetic_vendi_scores(
    *,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    diversity_types: Sequence[str] | None = None,
    n_genes: int = 128,
    n_control: int = 1024,
    n_per_perturbation: int = 1024,
    n_perturbations: int = 128,
    batch_size: int = 1024,
    n_pca_components: int = 50,
    sample_size: int = 2000,
    random_state: int = 0,
    by_context: bool = True,
) -> str:
    """
    Compute Vendi + PDS for CausalDGP under each diversity scenario.

    Args:
        output_dir: Directory where the result CSV should be written.
        diversity_types: Diversity types to generate, or ``None`` for all of them.
        n_genes: Number of genes to simulate.
        n_control: Number of control cells.
        n_per_perturbation: Number of cells per perturbation.
        n_perturbations: Number of perturbations.
        batch_size: Batch size passed to the Vendi scorer.
        n_pca_components: Number of PCA components used by the Vendi scorer.
        sample_size: Number of cells sampled for MMD bandwidth estimation.
        random_state: Random seed, also used as the CausalDGP generator seed.
        by_context: When True, score each context value separately.

    Returns:
        Path to the written result CSV.
    """
    if diversity_types is None:
        diversity_types = list(_DIVERSITY_TYPES)
    for dt in diversity_types:
        if dt not in _DIVERSITY_TYPES:
            raise ValueError(f"Unknown diversity_type={dt!r}. Choose from {_DIVERSITY_TYPES}.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / (
        f"synthetic_vendi_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    rows: list[dict[str, Any]] = []
    for dt in diversity_types:
        print(f"Computing scores for causalDGP diversity_type={dt}")
        spec_rows = _compute_synthetic_vendi_score(
            diversity_type=dt,
            n_genes=n_genes,
            n_control=n_control,
            n_per_perturbation=n_per_perturbation,
            n_perturbations=n_perturbations,
            batch_size=batch_size,
            n_pca_components=n_pca_components,
            sample_size=sample_size,
            random_state=random_state,
            by_context=by_context,
        )
        for row in spec_rows:
            rows.append(row)
            print(
                f"  [{row['scope']}] vendi_cell={row['vendi_score_cell']:.6g}, "
                f"pds_l1={row.get('pds_l1', float('nan')):.4f}, "
                f"cells={row['n_cells']}, perts={row['n_total_perturbations']}"
            )
        pd.DataFrame(rows, columns=_OUTPUT_COLUMNS).to_csv(output_path, index=False)

    print(f"Done. Synthetic Vendi scores saved to: {output_path}")
    return str(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute Vendi and split-half PDS scores for real and synthetic datasets."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="real",
        choices=["real", "synthetic", "both"],
        help="Score real datasets, synthetic CausalDGP datasets, or both.",
    )
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset_label",
        type=str,
        nargs="+",
        choices=list(_DATASET_SPEC_BY_LABEL),
        help="Dataset label(s) to evaluate. Omit to run all configured datasets.",
    )
    parser.add_argument(
        "--obs_layer",
        type=str,
        default=NORM_LAYER_KEY,
        help=(
            "Expression layer for Vendi/PDS, applied to both real and synthetic datasets. "
            "Set to 'none' to use adata.X (raw counts for CausalDGP)."
        ),
    )
    parser.add_argument(
        "--counts_layer",
        type=str,
        default="counts",
        help="Layer used only to build normalized_log1p for h5ad inputs. Set to 'none' to use adata.X.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_pca_components", type=int, default=50)
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--norm_target_sum", type=float, default=1e4)
    parser.add_argument(
        "--no_by_context",
        action="store_true",
        help="Score the whole dataset instead of each context value separately.",
    )
    # Synthetic CausalDGP options.
    parser.add_argument(
        "--diversity_type",
        type=str,
        nargs="+",
        choices=list(_DIVERSITY_TYPES),
        help="Diversity type(s) for CausalDGP. Omit to run all (A, b, both, none).",
    )
    parser.add_argument("--n_genes", type=int, default=128, help="Genes for CausalDGP.")
    parser.add_argument("--n_control", type=int, default=1024, help="Control cells for CausalDGP.")
    parser.add_argument(
        "--n_per_perturbation", type=int, default=1024, help="Cells per perturbation for CausalDGP."
    )
    parser.add_argument(
        "--n_perturbations", type=int, default=128, help="Number of perturbations for CausalDGP."
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    by_context = not args.no_by_context

    if args.source in ("real", "both"):
        run_real_dataset_vendi_scores(
            output_dir=args.output_dir,
            dataset_labels=args.dataset_label,
            obs_layer=_parse_optional_layer(args.obs_layer),
            counts_layer=_parse_optional_layer(args.counts_layer),
            batch_size=int(args.batch_size),
            n_pca_components=int(args.n_pca_components),
            sample_size=int(args.sample_size),
            random_state=int(args.random_state),
            norm_target_sum=float(args.norm_target_sum),
            by_context=by_context,
        )

    if args.source in ("synthetic", "both"):
        run_synthetic_vendi_scores(
            output_dir=args.output_dir,
            diversity_types=args.diversity_type,
            n_genes=int(args.n_genes),
            n_control=int(args.n_control),
            n_per_perturbation=int(args.n_per_perturbation),
            n_perturbations=int(args.n_perturbations),
            batch_size=int(args.batch_size),
            n_pca_components=int(args.n_pca_components),
            sample_size=int(args.sample_size),
            random_state=int(args.random_state),
            by_context=by_context,
        )


if __name__ == "__main__":
    main()
