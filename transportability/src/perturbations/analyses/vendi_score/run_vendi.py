"""Compute whole-dataset Vendi scores for the real perturbation datasets."""

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
import pandas as pd

from ...data.cd4_chunked import CD4ChunkedDataset
from ...metrics.reconstruction.vendi_score import vendi_score
from ..common import NORM_LAYER_KEY
from ..context import get_dataset_context_config
from ..util import (
    count_non_control_perturbations,
    ensure_normalized_log1p_layer,
    load_real_dataset,
    validate_perturbation_targets_subset_from_obs,
)

_DEFAULT_OUTPUT_DIR = "results/vendi_score"
_CONTROL_LABEL = "control"
_CD4_DATASET = "CD4+"
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
    "vendi_score",
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
        "vendi_score": score,
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
    obs_layer: str | None,
    counts_layer: str | None,
    batch_size: int,
    n_pca_components: int,
    sample_size: int,
    random_state: int,
    norm_target_sum: float,
) -> dict[str, Any]:
    """Load one in-memory h5ad dataset and compute its whole-dataset Vendi score."""
    _require_existing_path(spec.dataset_path)

    adata, _ = load_real_dataset(dataset_path=spec.dataset_path)
    try:
        if spec.dataset_variant is not None:
            _validate_replogle22_variant(adata, spec.dataset_variant)

        vendi_layer_key = _ensure_h5ad_layer(
            adata=adata,
            obs_layer=obs_layer,
            counts_layer=counts_layer,
            norm_target_sum=norm_target_sum,
        )
        validate_perturbation_targets_subset_from_obs(
            obs=adata.obs,
            gene_names=adata.var_names,
            control_label=_CONTROL_LABEL,
        )

        start_time = time.perf_counter()
        score = vendi_score(
            ac=adata,
            ac_batch_size=batch_size,
            n_pca_components=n_pca_components,
            sample_size=sample_size,
            random_state=random_state,
            layer_key=vendi_layer_key,
            control_label=_CONTROL_LABEL,
        )
        execution_time_seconds = time.perf_counter() - start_time

        return _base_result_row(
            spec=spec,
            obs=adata.obs,
            n_vars=adata.n_vars,
            reported_layer_key=obs_layer or "X",
            batch_size=batch_size,
            n_pca_components=n_pca_components,
            sample_size=sample_size,
            random_state=random_state,
            execution_time_seconds=execution_time_seconds,
            score=score,
        )
    finally:
        del adata
        gc.collect()


def _compute_cd4_vendi_score(
    *,
    spec: DatasetSpec,
    obs_layer: str | None,
    batch_size: int,
    n_pca_components: int,
    sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    """Compute whole-dataset Vendi for CD4+ using a backed AnnCollection."""
    _require_existing_path(spec.dataset_path)

    runtime = CD4ChunkedDataset.from_manifest(spec.dataset_path)
    vendi_layer_key = _cd4_vendi_layer(runtime, obs_layer)
    validate_perturbation_targets_subset_from_obs(
        obs=runtime.obs,
        gene_names=runtime.var_names,
        control_label=_CONTROL_LABEL,
    )

    with runtime.open_collection() as handle:
        start_time = time.perf_counter()
        score = vendi_score(
            ac=handle.collection,
            ac_batch_size=batch_size,
            n_pca_components=n_pca_components,
            sample_size=sample_size,
            random_state=random_state,
            layer_key=vendi_layer_key,
            control_label=_CONTROL_LABEL,
        )
        execution_time_seconds = time.perf_counter() - start_time

    try:
        return _base_result_row(
            spec=spec,
            obs=runtime.obs,
            n_vars=runtime.n_vars,
            reported_layer_key=obs_layer or "X",
            batch_size=batch_size,
            n_pca_components=n_pca_components,
            sample_size=sample_size,
            random_state=random_state,
            execution_time_seconds=execution_time_seconds,
            score=score,
        )
    finally:
        del runtime
        gc.collect()


def run_real_dataset_vendi_scores(
    *,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    dataset_labels: Sequence[str] | None = None,
    obs_layer: str | None = NORM_LAYER_KEY,
    counts_layer: str | None = "counts",
    batch_size: int = 1024,
    n_pca_components: int = 30,
    sample_size: int = 2000,
    random_state: int = 0,
    norm_target_sum: float = 1e4,
) -> str:
    """
    Compute whole-dataset Vendi scores for all configured real datasets.

    Args:
        output_dir: Directory where the result CSV should be written.
        dataset_labels: Dataset labels to evaluate, or ``None`` for all datasets.
        obs_layer: Expression layer to evaluate, or ``None`` to use ``.X``.
        counts_layer: Count layer used only when building ``normalized_log1p`` for h5ad inputs.
        batch_size: Batch size passed to the Vendi scorer.
        n_pca_components: Number of PCA components used by the Vendi scorer.
        sample_size: Number of cells sampled for MMD bandwidth estimation.
        random_state: Random seed for deterministic sampling.
        norm_target_sum: Library-size target sum when constructing ``normalized_log1p``.

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
            row = _compute_cd4_vendi_score(
                spec=spec,
                obs_layer=obs_layer,
                batch_size=int(batch_size),
                n_pca_components=int(n_pca_components),
                sample_size=int(sample_size),
                random_state=int(random_state),
            )
        else:
            row = _compute_h5ad_vendi_score(
                spec=spec,
                obs_layer=obs_layer,
                counts_layer=counts_layer,
                batch_size=int(batch_size),
                n_pca_components=int(n_pca_components),
                sample_size=int(sample_size),
                random_state=int(random_state),
                norm_target_sum=float(norm_target_sum),
            )
        rows.append(row)
        print(
            f"  vendi_score={row['vendi_score']:.6g}, "
            f"cells={row['n_cells']}, genes={row['n_genes']}, "
            f"contexts={row['n_contexts']} ({row['context_values']})"
        )
        pd.DataFrame(rows, columns=_OUTPUT_COLUMNS).to_csv(output_path, index=False)

    print(f"Done. Vendi scores saved to: {output_path}")
    return str(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute whole-dataset single-cell MMD Vendi scores for all real datasets."
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
        help="Expression layer for Vendi. Set to 'none' to use adata.X.",
    )
    parser.add_argument(
        "--counts_layer",
        type=str,
        default="counts",
        help="Layer used only to build normalized_log1p for h5ad inputs. Set to 'none' to use adata.X.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_pca_components", type=int, default=30)
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--norm_target_sum", type=float, default=1e4)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
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
    )


if __name__ == "__main__":
    main()
