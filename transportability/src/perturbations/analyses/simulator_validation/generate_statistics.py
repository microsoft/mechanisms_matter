r"""
Entry point: generate simulator-validation statistics for real or synthetic data.

Leverages the existing data paths used by the benchmark drivers:

- Real datasets are loaded with ``load_real_dataset`` (the same loader used by
  ``analyses/real_experiments/run.py``).
- Synthetic CausalDGP datasets are generated with ``causalDGP`` using the fitted
  Norman19 parameters (the same generator used by
  ``analyses/synthetic_simulations/random_sweep.py``).

For each dataset it computes the marginal/pairwise panel and the
TRADE perturbation-effect statistics, then writes median + bootstrap CI summary
tables.

Examples:
    # Real dataset
    python -m perturbations.analyses.simulator_validation.generate_statistics \\
        --source real --name norman19 \\
        --dataset-path data/norman19/norman19_processed.h5ad \\
        --counts-layer counts --output-dir results/simulator_validation

    # Synthetic CausalDGP (needs fitted parameters from parameter_estimation.py)
    python -m perturbations.analyses.simulator_validation.generate_statistics \\
        --source synthetic --name causalDGP \\
        --G 128 --P 128 --diversity-type both \\
        --output-dir results/simulator_validation
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from perturbations.analyses.common import NORM_LAYER_KEY
from perturbations.analyses.context import CONTEXT_DATA_MAP, get_dataset_context_config
from perturbations.analyses.synthetic_simulations.sampling import (
    load_parameter_estimation_inputs,
)
from perturbations.analyses.util import load_real_dataset
from perturbations.data.dgp import causalDGP
from perturbations.metrics.perturbation_effect.trade import (
    perturbation_effect_statistics,
    summarize_perturbation_statistics,
)
from perturbations.metrics.summary_statistics import (
    cell_wise_statistics,
    gene_pair_correlations,
    gene_wise_statistics,
    summarize_statistics,
)

_DEFAULT_OUTPUT_DIR = "results/simulator_validation"
_DEFAULT_CONTEXT_KEYS = tuple(
    dict.fromkeys(
        get_dataset_context_config(dataset_name).context_axis for dataset_name in CONTEXT_DATA_MAP
    )
)


def _none_if_empty(value: str | None) -> str | None:
    """Return ``None`` for empty/``"none"`` strings, else the value."""
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" or stripped.lower() == "none" else stripped


def resolve_context_key(adata: ad.AnnData, value: str | None = "auto") -> str:
    """Resolve an explicit context key or infer the dataset's standard context column."""
    requested = _none_if_empty(value)
    if requested is None:
        raise ValueError("Context-specific statistics require a context key.")
    if requested.lower() != "auto":
        if requested not in adata.obs.columns:
            raise KeyError(
                f"Context key {requested!r} not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return requested

    for key in _DEFAULT_CONTEXT_KEYS:
        if key in adata.obs.columns:
            return key

    raise KeyError(
        "Could not infer a context column. Expected one of "
        f"{list(_DEFAULT_CONTEXT_KEYS)} in adata.obs; pass --context-key explicitly."
    )


def _tag(df: pd.DataFrame, group: str, name: str, context: str = "all") -> pd.DataFrame:
    """Prepend ``group``/``dataset``/``context`` columns and normalize the label column."""
    out = df.copy()
    if "metric" in out.columns:
        out = out.rename(columns={"metric": "statistic"})
    out.insert(0, "group", group)
    out.insert(1, "dataset", name)
    out.insert(2, "context", context)
    return out


def _frac_abs_gt_summary(
    pairs: dict,
    threshold: float,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Summarize the fraction of gene pairs with ``|correlation| > threshold``.

    Signed correlations have a pooled median near zero, so co-expression strength
    is better captured by the fraction of strongly correlated pairs. Returns one
    row each for Pearson and Kendall with a bootstrap confidence interval.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for key, stat in (
        ("pearson_values", "frac_abs_pearson_gt"),
        ("kendall_values", "frac_abs_kendall_gt"),
    ):
        values = np.abs(np.asarray(pairs[key], dtype=np.float64).ravel())
        values = values[np.isfinite(values)]
        indicator = (values > threshold).astype(np.float64)
        frac = float(indicator.mean()) if indicator.size else float("nan")
        if indicator.size > 1:
            boot = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                boot[b] = indicator[rng.integers(0, indicator.size, indicator.size)].mean()
            alpha = 1.0 - confidence
            ci_low = float(np.quantile(boot, alpha / 2.0))
            ci_high = float(np.quantile(boot, 1.0 - alpha / 2.0))
        else:
            ci_low = ci_high = frac
        rows.append(
            {
                "statistic": f"{stat}_{threshold}",
                "median": frac,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(indicator.size),
                "confidence": confidence,
            }
        )
    return pd.DataFrame(
        rows, columns=["statistic", "median", "ci_low", "ci_high", "n", "confidence"]
    )


def load_synthetic_dataset(
    n_genes: int = 128,
    n_control: int = 1024,
    n_per_perturbation: int = 1024,
    n_perturbations: int = 128,
    diversity_type: str = "both",
    seed: int = 0,
) -> ad.AnnData:
    """
    Generate a CausalDGP dataset from fitted Norman19 parameters.

    Args:
        n_genes: Number of genes to simulate.
        n_control: Number of control cells.
        n_per_perturbation: Cells per perturbation.
        n_perturbations: Number of perturbations.
        diversity_type: Cross-context mechanism shift (``"A"``/``"b"``/``"both"``/
            ``"none"``).
        seed: RNG seed for the simulator.

    Returns:
        The simulated AnnData (raw counts in ``.X``, log-normalized values in the
        ``normalized_log1p`` layer, ``obs['perturbation']`` and
        ``obs['cell_line']``).
    """
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
        seed=seed,
        normalize=True,
        normalized_layer_key=NORM_LAYER_KEY,
    )
    return adata


def compute_validation_summary(
    adata: ad.AnnData,
    name: str,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    context_key: str | None = "auto",
    batch_key: str | None = None,
    counts_layer: str | None = None,
    lognorm_layer: str | None = None,
    max_genes: int = 200,
    max_cells: int = 2000,
    n_replicates: int = 3,
    min_cells_per_replicate: int = 10,
    deg_fdr: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    include_gene_pairs: bool = True,
    include_perturbation_effects: bool = True,
    context_workers: int = 2,
    metacell_size: int = 10,
    frac_threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Compute the full validation summary for one dataset.

    Every block is computed separately per context (e.g. per cell line), and
    each summary row is tagged with its context.

    The marginal (gene-wise, cell-wise) and gene-pair blocks are computed on the
    control cells so baseline realism is not confounded by perturbation effects.
    The perturbation-effect block uses all cells, since DESeq2 contrasts each
    perturbation against control within each context.

    Returns:
        Tuple ``(summary_df, perturbation_effect_df)``. ``summary_df`` holds
        median + CI rows tagged by ``group``/``dataset``/``context``; the second
        element is the per-perturbation table (or ``None`` if not computed).
    """
    context_key = resolve_context_key(adata, context_key)
    summaries: list[pd.DataFrame] = []
    context_values = list(pd.unique(np.asarray(adata.obs[context_key]).astype(str)))

    # Marginal and gene-pair blocks, per context and on control cells only.
    for ctx in context_values:
        sub = adata[np.asarray(adata.obs[context_key]).astype(str) == ctx]
        ctx_label = str(ctx)
        control_mask = np.asarray(sub.obs[perturbation_key]).astype(str) == control_label
        if not control_mask.any():
            raise ValueError(f"No control cells for {name!r} (context={ctx_label}).")
        control_sub = sub[control_mask]

        gene_wise = gene_wise_statistics(control_sub, layer=counts_layer)
        summaries.append(
            _tag(
                summarize_statistics(gene_wise, n_boot=n_boot, seed=seed),
                "gene_wise",
                name,
                ctx_label,
            )
        )

        cell_wise = cell_wise_statistics(control_sub, layer=counts_layer)
        summaries.append(
            _tag(
                summarize_statistics(cell_wise, n_boot=n_boot, seed=seed),
                "cell_wise",
                name,
                ctx_label,
            )
        )

        if include_gene_pairs:
            try:
                pairs = gene_pair_correlations(
                    control_sub,
                    layer=lognorm_layer,
                    max_genes=max_genes,
                    max_cells=max_cells,
                    seed=seed,
                    metacell_size=metacell_size,
                    counts_layer=counts_layer,
                )
                pair_df = pd.DataFrame(
                    {
                        "pearson": pairs["pearson_values"],
                        "kendall": pairs["kendall_values"],
                        "abs_pearson": np.abs(pairs["pearson_values"]),
                        "abs_kendall": np.abs(pairs["kendall_values"]),
                    }
                )
                summaries.append(
                    _tag(
                        summarize_statistics(pair_df, n_boot=n_boot, seed=seed),
                        "gene_pair",
                        name,
                        ctx_label,
                    )
                )
                summaries.append(
                    _tag(
                        _frac_abs_gt_summary(pairs, frac_threshold, n_boot, seed),
                        "gene_pair",
                        name,
                        ctx_label,
                    )
                )
            except (AssertionError, KeyError, ValueError) as error:
                warnings.warn(
                    f"Skipping gene-pair block for {name!r} (context={ctx_label}): {error}",
                    stacklevel=2,
                )

    # Perturbation-effect block: DESeq2 already runs within each context.
    perturbation_effect_df: pd.DataFrame | None = None
    if include_perturbation_effects:
        if batch_key is None:
            warnings.warn(
                f"No batch_key given for {name!r}; pseudobulk replicates will be "
                "formed by randomly splitting cells instead of real batches.",
                stacklevel=2,
            )
        try:
            perturbation_effect_df = perturbation_effect_statistics(
                adata,
                perturbation_key=perturbation_key,
                control_label=control_label,
                context_key=context_key,
                batch_key=batch_key,
                layer=counts_layer,
                n_replicates=n_replicates,
                min_cells_per_replicate=min_cells_per_replicate,
                deg_fdr=deg_fdr,
                seed=seed,
                context_workers=context_workers,
            )
            for ctx, group_df in perturbation_effect_df.groupby(context_key):
                summaries.append(
                    _tag(
                        summarize_perturbation_statistics(group_df, n_boot=n_boot, seed=seed),
                        "perturbation_effect",
                        name,
                        str(ctx),
                    )
                )
        except (ImportError, KeyError, ValueError) as error:
            warnings.warn(f"Skipping perturbation-effect block for {name!r}: {error}", stacklevel=2)

    return pd.concat(summaries, ignore_index=True), perturbation_effect_df


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate simulator-validation statistics for real or synthetic data."
    )
    parser.add_argument("--source", required=True, choices=["real", "synthetic"])
    parser.add_argument("--name", default=None, help="Dataset label (defaults from source).")

    # Real-data options.
    parser.add_argument("--dataset-path", default=None, help="Path to the real dataset .h5ad.")

    # Synthetic-data (CausalDGP) options.
    parser.add_argument("--G", type=int, default=128, help="Number of genes to simulate.")
    parser.add_argument("--N0", type=int, default=1024, help="Number of control cells.")
    parser.add_argument("--Nk", type=int, default=1024, help="Cells per perturbation.")
    parser.add_argument("--P", type=int, default=128, help="Number of perturbations.")
    parser.add_argument("--diversity-type", default="both", choices=["A", "B", "both", "none"])

    # Statistic options.
    parser.add_argument("--perturbation-key", default="perturbation")
    parser.add_argument("--control-label", default="control")
    parser.add_argument(
        "--context-key",
        default="auto",
        help="Context column (default: auto-detect cell_line, context, or donor_timepoint).",
    )
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--counts-layer", default=None)
    parser.add_argument("--lognorm-layer", default=None)
    parser.add_argument("--max-genes", type=int, default=200)
    parser.add_argument("--max-cells", type=int, default=2000)
    parser.add_argument("--n-replicates", type=int, default=3)
    parser.add_argument("--min-cells-per-replicate", type=int, default=10)
    parser.add_argument("--deg-fdr", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--context-workers",
        type=int,
        default=2,
        help="Maximum parallel TRADE context workers (default: 2).",
    )
    parser.add_argument("--skip-gene-pairs", action="store_true")
    parser.add_argument("--skip-perturbation-effects", action="store_true")
    parser.add_argument(
        "--metacell-size",
        type=int,
        default=10,
        help="Cells per metacell for gene-pair correlations (0 = single cells).",
    )
    parser.add_argument(
        "--frac-threshold",
        type=float,
        default=0.3,
        help="Report the fraction of gene pairs with |correlation| above this value.",
    )
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    return parser


def _load_dataset(
    args: argparse.Namespace,
) -> tuple[ad.AnnData, str, str | None, str | None, str | None]:
    """Load the dataset and resolve source-specific layer/context/batch defaults."""
    context_key = _none_if_empty(args.context_key)
    batch_key = _none_if_empty(args.batch_key)
    counts_layer = _none_if_empty(args.counts_layer)
    lognorm_layer = _none_if_empty(args.lognorm_layer)

    if args.source == "synthetic":
        name = args.name or "causalDGP"
        print(f"Generating CausalDGP (G={args.G}, P={args.P}, diversity={args.diversity_type}) ...")
        adata = load_synthetic_dataset(
            n_genes=args.G,
            n_control=args.N0,
            n_per_perturbation=args.Nk,
            n_perturbations=args.P,
            diversity_type=args.diversity_type,
            seed=args.seed,
        )
        # CausalDGP defaults: counts in .X and a log-normalized layer.
        if lognorm_layer is None:
            lognorm_layer = NORM_LAYER_KEY
    else:
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required when --source real.")
        name = args.name or Path(args.dataset_path).stem
        print(f"Loading real dataset {args.dataset_path} ...")
        adata, _ = load_real_dataset(dataset_path=args.dataset_path)
        # Real processed datasets keep raw counts in a 'counts' layer by default.
        if counts_layer is None and "counts" in getattr(adata, "layers", {}):
            counts_layer = "counts"

    context_key = resolve_context_key(adata, context_key)
    args.context_key = context_key
    args.batch_key = batch_key
    args.counts_layer = counts_layer
    args.lognorm_layer = lognorm_layer
    return adata, name, context_key, counts_layer, batch_key


def main() -> None:
    """Parse arguments, load/generate a dataset, and write summary tables."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata, name, context_key, counts_layer, batch_key = _load_dataset(args)
    print(f"Dataset {name!r}: {adata.n_obs} cells x {adata.n_vars} genes.")

    summary_df, perturbation_effect_df = compute_validation_summary(
        adata,
        name=name,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        context_key=context_key,
        batch_key=batch_key,
        counts_layer=counts_layer,
        lognorm_layer=args.lognorm_layer,
        max_genes=args.max_genes,
        max_cells=args.max_cells,
        n_replicates=args.n_replicates,
        min_cells_per_replicate=args.min_cells_per_replicate,
        deg_fdr=args.deg_fdr,
        n_boot=args.n_boot,
        seed=args.seed,
        include_gene_pairs=not args.skip_gene_pairs,
        include_perturbation_effects=not args.skip_perturbation_effects,
        context_workers=args.context_workers,
        metacell_size=args.metacell_size,
        frac_threshold=args.frac_threshold,
    )

    summary_path = output_dir / f"{name}_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary to {summary_path}")
    print(summary_df.to_string(index=False))

    if perturbation_effect_df is not None:
        effect_path = output_dir / f"{name}_perturbation_effect.csv"
        perturbation_effect_df.to_csv(effect_path, index=False)
        print(f"\nWrote per-perturbation effects to {effect_path}")


if __name__ == "__main__":
    main()
