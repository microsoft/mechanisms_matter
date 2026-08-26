"""Create publication-style boxplots for real experiment benchmark results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..plot_utils import (
    COMMON_METRIC_BASE_COLORS,
    COMMON_METRIC_LABELS,
    MODEL_TICK_LABEL_ALIGNMENT,
    MODEL_TICK_LABEL_ROTATION,
    apply_paper_plot_style,
    coerce_numeric,
    compute_metric_limits,
    has_valid_metric,
    resolve_expected_model_order,
    save_metric_group_boxplot,
    with_vendi_ratio,
)

apply_paper_plot_style()

PAIR_METRIC_GROUPS: tuple[tuple[str, str], ...] = (
    ("pearson", "pearson_degs"),
    ("mae", "mae_degs"),
    ("mse", "mse_degs"),
    ("r2", "r2_degs"),
)
DES_METRICS: tuple[str, ...] = ("des_recall", "des_precision", "des_jaccard")
SINGLE_METRICS: tuple[str, ...] = (
    "parametric_distance",
    "mmd_distance",
    "fid_distance",
    *DES_METRICS,
)
PDS_METRICS: tuple[str, ...] = ("pds_l1", "pds_l2", "pds_cosine")
PLOT_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *[metric for pair in PAIR_METRIC_GROUPS for metric in pair],
            *SINGLE_METRICS,
            *PDS_METRICS,
            "vendi_ratio",
        )
    )
)
NUMERIC_RESULT_COLUMNS: tuple[str, ...] = (
    *PLOT_METRICS,
    "vendi_score_pred",
    "vendi_score_obs",
)

METRIC_LABELS: dict[str, str] = COMMON_METRIC_LABELS
METRIC_BASE_COLORS: dict[str, str] = COMMON_METRIC_BASE_COLORS


def latest_results_file(search_dir: Path) -> Path:
    """Return the newest results CSV in the given directory."""
    files = sorted(search_dir.glob("real_experiment_results_*.csv"))
    if not files:
        raise FileNotFoundError(f"No result files found in: {search_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_results_path(results_arg: str | None) -> Path:
    """Resolve an explicit results path or discover the latest default results file."""
    if results_arg:
        path = Path(results_arg)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        return path
    return latest_results_file(Path("results/real_experiments"))


def prepare_results_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate raw results before plotting."""
    prepared = df.copy()
    prepared["model"] = prepared["model"].astype("string").str.strip()
    before = len(prepared)
    prepared = prepared[prepared["status"].astype(str).str.lower() == "success"].copy()
    print(f"Kept successful runs: {len(prepared)}/{before}")

    coerce_numeric(prepared, NUMERIC_RESULT_COLUMNS)

    if prepared.empty:
        raise ValueError("No rows available to plot after filtering.")
    return prepared


def filter_context(df: pd.DataFrame, context_value: str, *, label: str = "") -> pd.DataFrame:
    """Filter rows by the requested context value."""
    before = len(df)
    filtered = df[df["context_values"] == context_value].copy()
    context_label = f"{label} " if label else ""
    print(
        f"Filtered {context_label}to context_values == '{context_value}': {len(filtered)}/{before} rows"
    )
    if filtered.empty:
        raise ValueError(f"No rows found for context_values == '{context_value}'.")
    return filtered


def save_boxplot(
    df: pd.DataFrame,
    model_order: Sequence[str],
    metrics: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
    cluster_gap: float,
    dpi: int,
    metric_limits: dict[str, tuple[float, float]] | None = None,
) -> bool:
    """Save one grouped boxplot with the shared real-experiment styling."""
    return save_metric_group_boxplot(
        data=df,
        model_order=model_order,
        metrics=metrics,
        output_path=output_path,
        title=title,
        ylabel=ylabel,
        metric_labels=METRIC_LABELS,
        metric_base_colors=METRIC_BASE_COLORS,
        cluster_gap=cluster_gap,
        fig_size=(8.0, 5.8),
        metric_limits=metric_limits,
        dpi=dpi,
        no_model_message=None,
        x_tick_label_rotation=MODEL_TICK_LABEL_ROTATION,
        x_tick_label_ha=MODEL_TICK_LABEL_ALIGNMENT,
    )


def main() -> None:
    """Parse CLI arguments, load data, and save all real-experiment plots."""
    parser = argparse.ArgumentParser(
        description="Plot model-comparison boxplots for real experiment results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Path to a results CSV. If omitted, the latest file in results/real_experiments is used."
        ),
    )
    parser.add_argument(
        "--results_compare",
        type=str,
        default=None,
        help="Path to a second results CSV to compare (optional). If provided, the y-axis range will be determined by both datasets.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved files (default: 300).",
    )
    parser.add_argument(
        "--cluster_gap",
        type=float,
        default=1.2,
        help="Horizontal gap between model clusters (default: 1.2).",
    )
    parser.add_argument(
        "--context_value",
        type=str,
        default="K562",
        help="Filter results to this context value if 'context_values' column is present (default: 'K562').",
    )
    args = parser.parse_args()

    results_path = resolve_results_path(args.results)
    print(f"Using results: {results_path}")

    df = filter_context(prepare_results_data(pd.read_csv(results_path)), args.context_value)

    model_order = resolve_expected_model_order(df, require_all_expected=True)
    print(f"Models: {model_order}")

    metric_limits: dict[str, tuple[float, float]] | None = None
    if args.results_compare:
        compare_path = resolve_results_path(args.results_compare)
        print(f"Using comparison results for y-axis limits: {compare_path}")

        compare_df = filter_context(
            prepare_results_data(pd.read_csv(compare_path)),
            args.context_value,
            label="comparison",
        )
        metric_limits = compute_metric_limits(
            [with_vendi_ratio(results) for results in (df, compare_df)],
            PLOT_METRICS,
        )

    output_dir = results_path.parent / f"{results_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_gap = float(args.cluster_gap)
    dpi = int(args.dpi)

    for left_metric, right_metric in PAIR_METRIC_GROUPS:
        metrics = [metric for metric in (left_metric, right_metric) if has_valid_metric(df, metric)]
        if not metrics:
            print(f"Skipping {(left_metric, right_metric)}: no valid values.")
            continue

        title = (
            f"{METRIC_LABELS[metrics[0]]} vs {METRIC_LABELS[metrics[1]]}"
            if len(metrics) == 2
            else METRIC_LABELS[metrics[0]]
        )
        output_path = output_dir / f"{left_metric}_vs_{right_metric}_boxplot.svg"
        if save_boxplot(
            df=df,
            model_order=model_order,
            metrics=metrics,
            output_path=output_path,
            title=title,
            ylabel="Score",
            cluster_gap=cluster_gap,
            dpi=dpi,
            metric_limits=metric_limits,
        ):
            print(f"Saved: {output_path}")

    for metric in SINGLE_METRICS:
        if not has_valid_metric(df, metric):
            print(f"Skipping {metric}: no valid values.")
            continue

        output_path = output_dir / f"{metric}_boxplot.svg"
        if save_boxplot(
            df=df,
            model_order=model_order,
            metrics=[metric],
            output_path=output_path,
            title=f"{METRIC_LABELS[metric]} by Model",
            ylabel="Distance" if metric.endswith("_distance") else "Score",
            cluster_gap=cluster_gap,
            dpi=dpi,
            metric_limits=metric_limits,
        ):
            print(f"Saved: {output_path}")

    vendi_df = with_vendi_ratio(df)
    vendi_output_path = output_dir / "vendi_ratio_boxplot.svg"
    if save_boxplot(
        df=vendi_df,
        model_order=model_order,
        metrics=["vendi_ratio"],
        output_path=vendi_output_path,
        title="Vendi Score Ratio by Model",
        ylabel="Ratio",
        cluster_gap=cluster_gap,
        dpi=dpi,
        metric_limits=metric_limits,
    ):
        print(f"Saved: {vendi_output_path}")
    else:
        print("Skipping vendi ratio plot: no valid values.")

    pds_metrics = [metric for metric in PDS_METRICS if has_valid_metric(df, metric)]
    if pds_metrics:
        pds_output_path = output_dir / "pds_metrics_boxplots.svg"
        if save_boxplot(
            df=df,
            model_order=model_order,
            metrics=pds_metrics,
            output_path=pds_output_path,
            title="PDS Metrics by Model",
            ylabel="Score",
            cluster_gap=cluster_gap,
            dpi=dpi,
            metric_limits=metric_limits,
        ):
            print(f"Saved: {pds_output_path}")
    else:
        print("Skipping PDS plot: no valid PDS metrics found.")


if __name__ == "__main__":
    main()
