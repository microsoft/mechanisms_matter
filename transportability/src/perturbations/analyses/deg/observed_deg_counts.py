"""Count observed DEG genes used by evaluation on held-out test data."""

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ...data.cd4_chunked import CD4ChunkedDataset
from ...data.dgp import causalDGP, directDGP
from ...metrics.gene_selection.differential_expression_score import (
    de_table_to_deg_masks,
    scanpy_de_table,
)
from ..common import NORM_LAYER_KEY
from ..context import (
    DEFAULT_CONTEXT_AXIS,
    ContextSplitter,
    DatasetContextConfig,
    build_evaluation_contexts,
    filter_indices_to_context_values,
    get_dataset_context_config,
)
from ..evaluator import get_eval_perturbation_ids
from ..synthetic_simulations.sampling import (
    ALL_PARAMS_PATH,
    PARAM_RANGES,
    load_parameter_estimation_inputs,
    sample_parameters,
)
from ..util import (
    build_perturbation_id_map,
    count_non_control_perturbations,
    ensure_normalized_log1p_layer,
    load_real_dataset,
    true_degs_for_context,
    validate_perturbation_targets_subset,
    validate_perturbation_targets_subset_from_obs,
)

_OUTPUT_DIR = "results/observed_deg_counts"
_REAL_DATASETS = ("norman19", "replogle22", "CD4+")
_SYNTHETIC_DATASETS = ("directDGP", "causalDGP")
_DATASET_CHOICES = (*_REAL_DATASETS, *_SYNTHETIC_DATASETS)


def _dataset_context_config(dataset_name: str) -> DatasetContextConfig:
    """Return the split-context configuration for one dataset."""
    if dataset_name in _REAL_DATASETS:
        return get_dataset_context_config(dataset_name)
    if dataset_name == "causalDGP":
        return DatasetContextConfig(
            context_axis=DEFAULT_CONTEXT_AXIS,
            heldout_values=(1,),
        )
    return DatasetContextConfig(context_axis=DEFAULT_CONTEXT_AXIS, heldout_values=None)


def _prepare_real_dataset(
    dataset_name: str,
    dataset_path: str,
    counts_layer: str | None,
    obs_layer: str | None,
    norm_target_sum: float,
) -> tuple[ad.AnnData | CD4ChunkedDataset, str | None]:
    """Load a real dataset and return the object plus the DEG layer to evaluate."""
    if Path(dataset_path).suffix.lower() == ".json":
        if dataset_name != "CD4+":
            raise ValueError("Chunked JSON manifests are only supported for dataset_name='CD4+'.")

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

        validate_perturbation_targets_subset_from_obs(
            obs=runtime.obs,
            gene_names=runtime.var_names,
            control_label="control",
        )
        return runtime, None

    adata, _ = load_real_dataset(dataset_path=dataset_path)

    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(
            f"Requested counts_layer='{counts_layer}' not found. Available layers: {list(adata.layers.keys())}"
        )

    if obs_layer is not None and obs_layer not in adata.layers:
        if obs_layer != NORM_LAYER_KEY:
            raise KeyError(
                f"Requested obs_layer='{obs_layer}' not found. Available layers: {list(adata.layers.keys())}"
            )
        source_layer = counts_layer or ("counts" if "counts" in adata.layers else None)
        ensure_normalized_log1p_layer(
            adata=adata,
            output_layer_key=obs_layer,
            source_layer=source_layer,
            target_sum=norm_target_sum,
        )

    validate_perturbation_targets_subset(adata=adata, control_label="control")
    return adata, obs_layer


def _load_synthetic_parameter_inputs() -> dict[str, np.ndarray]:
    """Load fitted parameter-estimation arrays used by the synthetic generators."""
    return load_parameter_estimation_inputs()


def _prepare_synthetic_dataset(
    dataset_name: str,
    sampled_params: dict[str, Any],
    synthetic_inputs: dict[str, np.ndarray],
    diversity_type: str,
    trial_id: int,
    obs_layer: str | None,
) -> tuple[ad.AnnData, Any]:
    """Generate one synthetic AnnData object and its true affected-gene masks."""
    common_kwargs = {
        "G": int(sampled_params["G"]),
        "N0": int(sampled_params["N0"]),
        "Nk": int(sampled_params["Nk"]),
        "P": int(sampled_params["P"]),
        "mu_l": float(sampled_params["mu_l"]),
        "all_theta": synthetic_inputs["all_theta"],
        "gene_names": synthetic_inputs["gene_names"],
        "seed": trial_id,
        "normalize": True,
        "normalized_layer_key": NORM_LAYER_KEY,
    }
    if dataset_name == "directDGP":
        adata, affected_genes = directDGP(
            **common_kwargs,
            p_effect=float(sampled_params["p_effect"]),
            effect_factor=float(sampled_params["effect_factor"]),
            B=float(sampled_params["B"]),
            control_mu=synthetic_inputs["control_mu"],
            pert_mu=synthetic_inputs["pert_mu"],
        )
    else:
        adata, affected_genes = causalDGP(
            **common_kwargs,
            diversity_type=diversity_type,
            verbose=True,
        )

    if obs_layer is not None and obs_layer not in adata.layers:
        raise KeyError(
            f"Requested obs_layer='{obs_layer}' not found. Available layers: {list(adata.layers.keys())}"
        )
    return adata, affected_genes


def _effective_split_strategy(
    adata: ad.AnnData | CD4ChunkedDataset,
    context_axis: str,
    split_strategy: str,
) -> str:
    """Return the split strategy after handling single-context datasets."""
    if split_strategy != "cross-context":
        return split_strategy

    if int(adata.obs[context_axis].nunique()) > 1:
        return split_strategy

    print(
        f"Warning: 'cross-context' split strategy is not applicable when "
        f"'{context_axis}' has only one value. Defaulting to 'in-context'."
    )
    return "in-context"


def _print_dataset_summary(
    adata: ad.AnnData | CD4ChunkedDataset,
    context_axis: str,
    dataset_name: str,
    dataset_path: str | None,
) -> None:
    """Print a compact dataset summary."""
    location = f" from {dataset_path}" if dataset_path else ""
    print(f"Loaded dataset '{dataset_name}'{location}")
    print(f"Shape: cells={adata.n_obs}, genes={adata.n_vars}")
    print(f"Contexts: {int(adata.obs[context_axis].nunique())} unique ({context_axis})")
    print(f"Perturbations (non-control): {count_non_control_perturbations(adata.obs)}")


def _count_observed_degs_for_trial(
    adata: ad.AnnData,
    splitter: ContextSplitter,
    trial_id: int,
    obs_layer: str | None,
    dataset_name: str | None = None,
    affected_genes: Any | None = None,
    fdr_threshold: float = 0.05,
    test_idx: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return per-perturbation observed DEG counts for one trial."""
    _, _, default_test_idx = splitter.split(seed=trial_id)
    if test_idx is None:
        test_idx = default_test_idx
    else:
        test_idx = np.asarray(test_idx, dtype=np.int64)
    labels = adata.obs["perturbation"].to_numpy(copy=False)
    control_test_idx = test_idx[labels[test_idx] == "control"]
    pert_label_to_id = None

    if affected_genes is not None:
        if dataset_name is None:
            raise ValueError("dataset_name is required when affected_genes are provided.")
        pert_label_to_id = build_perturbation_id_map(adata.obs)

    rows: list[dict[str, Any]] = []
    for eval_context in build_evaluation_contexts(splitter):
        bucket_test_idx = filter_indices_to_context_values(
            obs=adata.obs,
            indices=test_idx,
            context_axis=eval_context.axis,
            context_values=eval_context.values,
        )
        bucket_control_idx = filter_indices_to_context_values(
            obs=adata.obs,
            indices=control_test_idx,
            context_axis=eval_context.axis,
            context_values=eval_context.values,
        )
        if bucket_test_idx.size == 0 or bucket_control_idx.size == 0:
            continue

        bucket_obs_eval = adata[bucket_test_idx, :]
        bucket_perturbation_ids = get_eval_perturbation_ids(
            obs=bucket_obs_eval,
            control_label="control",
            strict_match=False,
        )
        if bucket_perturbation_ids.size == 0:
            continue

        bucket_degs = de_table_to_deg_masks(
            de_table=scanpy_de_table(
                adata=bucket_obs_eval.copy(),
                pert_col="perturbation",
                control_pert="control",
                key_added="test_de",
                layer=obs_layer,
            ),
            gene_names=adata.var_names,
            perturbation_ids=bucket_perturbation_ids,
            fdr_threshold=fdr_threshold,
        )
        if affected_genes is None:
            bucket_truth_stats: list[tuple[int | None, int | None]] = [(None, None)] * len(
                bucket_degs
            )
        else:
            if pert_label_to_id is None or dataset_name is None:
                raise RuntimeError("Synthetic truth mapping is not initialized.")
            true_deg_masks = true_degs_for_context(
                dataset_name=dataset_name,
                affected_genes=affected_genes,
                pert_label_to_id=pert_label_to_id,
                perturbation_ids=bucket_perturbation_ids,
                context_axis=eval_context.axis,
                context_values=eval_context.values,
            )
            bucket_truth_stats = [
                (
                    int(np.sum(observed_deg_mask & true_deg_mask)),
                    int(np.sum(true_deg_mask)),
                )
                for observed_deg_mask, true_deg_mask in zip(
                    bucket_degs,
                    true_deg_masks,
                    strict=True,
                )
            ]
        rows.extend(
            {
                "trial_id": int(trial_id),
                "context_axis": eval_context.axis,
                "context_values": eval_context.value_label,
                "perturbation": str(pert_id),
                "n_obs_degs": np.sum(deg_mask),
                "n_obs_degs_in_truth": n_obs_degs_in_truth,
                "n_deg_truth": n_deg_truth,
            }
            for pert_id, deg_mask, (n_obs_degs_in_truth, n_deg_truth) in zip(
                bucket_perturbation_ids,
                bucket_degs,
                bucket_truth_stats,
                strict=True,
            )
        )

    if not rows:
        raise ValueError(
            "No evaluation contexts contained both held-out controls and non-control perturbations."
        )
    return rows


def _count_observed_degs_for_cd4_trial(
    runtime: CD4ChunkedDataset,
    splitter: ContextSplitter,
    trial_id: int,
    obs_layer: str | None,
) -> list[dict[str, Any]]:
    """Materialize one CD4 test split and count observed DEGs on the subset."""
    _, _, test_idx = splitter.split(seed=trial_id)
    test_adata = runtime.materialize_subset(
        indices=test_idx,
        x_layer=obs_layer,
        include_layers=(),
    )
    try:
        return _count_observed_degs_for_trial(
            adata=test_adata,
            splitter=splitter,
            trial_id=trial_id,
            obs_layer=None,
            test_idx=np.arange(test_adata.n_obs, dtype=np.int64),
        )
    finally:
        del test_adata


def count_observed_degs(
    dataset_name: str,
    output_dir: str,
    n_trials: int,
    counts_layer: str | None = "counts",
    obs_layer: str | None = NORM_LAYER_KEY,
    split_strategy: str = "in-context",
    norm_target_sum: float = 1e4,
    dataset_path: str = "data/norman19/norman19_processed.h5ad",
    seed: int = 42,
    diversity_type: str = "both",
) -> str:
    """
    Count observed DEG genes in held-out test data and write a CSV.

    Args:
        dataset_name: Dataset name for either a real or synthetic dataset.
        output_dir: Directory where the output CSV should be written.
        n_trials: Number of trials to evaluate.
        counts_layer: Real-dataset count layer, or ``None`` to use ``adata.X``.
        obs_layer: Layer used for DEG computation, or ``None`` to use ``adata.X``.
        split_strategy: Evaluation split strategy.
        norm_target_sum: Normalization target sum when a real ``obs_layer`` must be built.
        dataset_path: Path to the processed real-dataset ``.h5ad`` file or CD4+ manifest ``.json``.
        seed: Master RNG seed for synthetic parameter sampling.
        diversity_type: Diversity mode for ``causalDGP``.

    Returns:
        The path to the written CSV file.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir_path / (
        f"observed_deg_counts_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    context_config = _dataset_context_config(dataset_name)

    all_rows: list[dict[str, Any]] = []

    if dataset_name in _REAL_DATASETS:
        real_data, eval_obs_layer = _prepare_real_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            counts_layer=counts_layer,
            obs_layer=obs_layer,
            norm_target_sum=norm_target_sum,
        )
        effective_split_strategy = _effective_split_strategy(
            adata=real_data,
            context_axis=context_config.context_axis,
            split_strategy=split_strategy,
        )
        _print_dataset_summary(
            adata=real_data,
            context_axis=context_config.context_axis,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )
        print(f"Counting observed DEGs for {n_trials} trial(s).")
        splitter = ContextSplitter(
            adata=real_data,
            split_strategy=effective_split_strategy,
            context_axis=context_config.context_axis,
            test_context_values=context_config.heldout_values,
        )
        row_metadata = {
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "split_strategy": effective_split_strategy,
            "n_cells": int(real_data.n_obs),
            "n_genes": int(real_data.n_vars),
        }
        for trial_id in range(int(n_trials)):
            print(f"Trial {trial_id + 1}/{n_trials}")
            if isinstance(real_data, CD4ChunkedDataset):
                trial_rows = _count_observed_degs_for_cd4_trial(
                    runtime=real_data,
                    splitter=splitter,
                    trial_id=trial_id,
                    obs_layer=obs_layer,
                )
            else:
                trial_rows = _count_observed_degs_for_trial(
                    adata=real_data,
                    splitter=splitter,
                    trial_id=trial_id,
                    obs_layer=eval_obs_layer,
                    dataset_name=dataset_name,
                )
            for row in trial_rows:
                row.update(row_metadata)
            all_rows.extend(trial_rows)
    elif dataset_name in _SYNTHETIC_DATASETS:
        synthetic_inputs = _load_synthetic_parameter_inputs()
        rng = np.random.default_rng(seed)

        print(f"Using synthetic parameter estimates from '{ALL_PARAMS_PATH}'.")
        print(f"Counting observed DEGs for {n_trials} synthetic trial(s).")

        for trial_id in range(int(n_trials)):
            print(f"Trial {trial_id + 1}/{n_trials}")
            sampled_params = sample_parameters(PARAM_RANGES, rng)
            adata, affected_genes = _prepare_synthetic_dataset(
                dataset_name=dataset_name,
                sampled_params=sampled_params,
                synthetic_inputs=synthetic_inputs,
                diversity_type=diversity_type,
                trial_id=trial_id,
                obs_layer=obs_layer,
            )
            effective_split_strategy = _effective_split_strategy(
                adata=adata,
                context_axis=context_config.context_axis,
                split_strategy=split_strategy,
            )
            _print_dataset_summary(
                adata=adata,
                context_axis=context_config.context_axis,
                dataset_name=dataset_name,
                dataset_path=None,
            )
            splitter = ContextSplitter(
                adata=adata,
                split_strategy=effective_split_strategy,
                context_axis=context_config.context_axis,
                test_context_values=context_config.heldout_values,
            )
            trial_rows = _count_observed_degs_for_trial(
                adata=adata,
                splitter=splitter,
                trial_id=trial_id,
                obs_layer=obs_layer,
                dataset_name=dataset_name,
                affected_genes=affected_genes,
            )
            for row in trial_rows:
                row.update(
                    {
                        "dataset": dataset_name,
                        "dataset_path": None,
                        "split_strategy": effective_split_strategy,
                        "n_cells": int(adata.n_obs),
                        "n_genes": int(adata.n_vars),
                        **sampled_params,
                        "diversity_type": diversity_type,
                        "seed": int(seed),
                    }
                )
            all_rows.extend(trial_rows)
    else:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name}. Expected one of {_DATASET_CHOICES}."
        )

    pd.DataFrame(all_rows).to_csv(csv_file, index=False)
    print(f"Done. Observed DEG counts saved to: {csv_file}")
    return str(csv_file)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a parser for real and synthetic observed-DEG counting."""
    parser = argparse.ArgumentParser(description="Count observed DEG genes used by evaluation.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="norman19",
        choices=_DATASET_CHOICES,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/norman19/norman19_processed.h5ad",
        help=(
            "Path to a processed real-dataset .h5ad file or a CD4+ chunk manifest .json. "
            "Ignored for synthetic datasets."
        ),
    )
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument(
        "--counts_layer",
        type=str,
        default="counts",
        help="Real-dataset count layer. Set to 'none' to use adata.X.",
    )
    parser.add_argument(
        "--obs_layer",
        type=str,
        default=NORM_LAYER_KEY,
        help="Layer to use for observed DEG computation. Set to 'none' to use adata.X.",
    )
    parser.add_argument(
        "--norm_target_sum",
        type=float,
        default=1e4,
        help="Target sum for normalization when a real obs_layer must be created.",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="in-context",
        choices=["in-context", "cross-context"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed for synthetic parameter sampling.",
    )
    parser.add_argument(
        "--diversity_type",
        type=str,
        default="both",
        choices=["A", "b", "both", "none"],
        help="Diversity mode for causalDGP.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    return build_arg_parser().parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    count_observed_degs(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_dir=_OUTPUT_DIR,
        n_trials=int(args.n_trials),
        counts_layer=(None if str(args.counts_layer).lower() == "none" else args.counts_layer),
        obs_layer=None if str(args.obs_layer).lower() == "none" else args.obs_layer,
        split_strategy=args.split_strategy,
        norm_target_sum=float(args.norm_target_sum),
        seed=int(args.seed),
        diversity_type=args.diversity_type,
    )


if __name__ == "__main__":
    main()
