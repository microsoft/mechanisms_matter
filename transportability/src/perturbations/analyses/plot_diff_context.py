"""Compare in-context and cross-context result metrics with grouped boxplots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .plot_utils import (
    COMMON_METRIC_LABELS,
    MODEL_TICK_LABEL_ALIGNMENT,
    MODEL_TICK_LABEL_ROTATION,
    SCLDM_OMEGA_MODEL_ORDER,
    apply_paper_plot_style,
    coerce_numeric,
    compute_metric_limits,
    filter_scldm_omega_results,
    metric_axis_label,
    save_metric_group_boxplot,
    transform_metric_values,
    with_vendi_ratio,
)

PLOT_CONTEXT_COLUMNS: tuple[str, str] = ("in_context", "cross_context")
METRICS: tuple[str, ...] = (
    "pearson",
    "pearson_true_degs",
    "pearson_degs",
    "mae",
    "mae_true_degs",
    "mae_degs",
    "mse",
    "mse_true_degs",
    "mse_degs",
    "r2",
    "r2_true_degs",
    "r2_degs",
    "parametric_distance",
    "mmd_distance",
    "fid_distance",
    "vendi_score_ratio",
    "des_recall",
    "des_precision",
    "des_jaccard",
    "pds_l1",
    "pds_l2",
    "pds_cosine",
)
RAW_RESULT_METRICS: tuple[str, ...] = tuple(
    metric for metric in METRICS if metric != "vendi_score_ratio"
)
NUMERIC_RESULT_COLUMNS: tuple[str, ...] = (
    *RAW_RESULT_METRICS,
    "vendi_score_pred",
    "vendi_score_obs",
)
METRIC_LABELS: dict[str, str] = {
    **COMMON_METRIC_LABELS,
    "pearson_true_degs": "Pearson (True DEGs)",
    "mae_true_degs": "MAE (True DEGs)",
    "mse_true_degs": "MSE (True DEGs)",
    "r2_true_degs": "R2 (True DEGs)",
}
CONTEXT_LABELS: dict[str, str] = {
    "in_context": "In-context",
    "cross_context": "Cross-context",
}
CONTEXT_COLORS: dict[str, str] = {
    "in_context": "#c92a2a",
    "cross_context": "#1c7ed6",
}

apply_paper_plot_style()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-metric in-context vs cross-context model comparisons, "
            "optionally for a single context value."
        )
    )
    parser.add_argument(
        "--in_context_results",
        type=str,
        required=True,
        help="Path to the in-context results CSV.",
    )
    parser.add_argument(
        "--cross_context_results",
        type=str,
        required=True,
        help="Path to the cross-context results CSV.",
    )
    parser.add_argument(
        "--context_value",
        type=str,
        default=None,
        help=(
            "Filter both results tables to rows whose 'context_values' column matches this value."
        ),
    )
    return parser.parse_args(argv)


def resolve_results_path(results_path: str) -> Path:
    """Resolve a results CSV path and validate that it exists."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return path


def prepare_results_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter successful rows and coerce all plotted metrics to numeric."""
    required_columns = {"model", "status"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Results table is missing required columns: {missing_columns}")

    prepared = data.copy()
    prepared["model"] = prepared["model"].astype("string").str.strip()

    before = len(prepared)
    prepared = prepared[prepared["status"].astype(str).str.lower() == "success"].copy()
    print(f"Kept successful runs: {len(prepared)}/{before}")
    if prepared.empty:
        raise ValueError("No successful rows available to plot.")

    for column in NUMERIC_RESULT_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = np.nan

    coerce_numeric(prepared, NUMERIC_RESULT_COLUMNS)
    prepared = with_vendi_ratio(prepared, ratio_column="vendi_score_ratio")
    if "vendi_score_ratio" not in prepared.columns:
        prepared["vendi_score_ratio"] = np.nan
    prepared = filter_scldm_omega_results(prepared)
    print(f"Kept scLDM omega rows: {len(prepared)}")
    return prepared


def filter_context_value(
    data: pd.DataFrame, context_value: str | None, *, label: str = ""
) -> pd.DataFrame:
    """Filter rows to a single context value when requested."""
    if context_value is None:
        return data
    context_value = context_value.strip()
    if not context_value:
        raise ValueError("context_value must be a non-empty string.")

    context_column = "context_values"
    if context_column not in data.columns:
        raise ValueError(
            "Cannot filter by context_value because the results table is missing "
            f"the '{context_column}' column."
        )

    before = len(data)
    filtered = data[data[context_column].astype("string").str.strip() == context_value].copy()
    label_prefix = f"{label} " if label else ""
    print(
        f"Filtered {label_prefix}to context_values == '{context_value}': "
        f"{len(filtered)}/{before} rows"
    )
    if filtered.empty:
        available_values = sorted(data[context_column].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows found for context_values == '{context_value}'. "
            f"Available values: {available_values}"
        )
    return filtered


def build_metric_comparison_data(
    in_context_data: pd.DataFrame,
    cross_context_data: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Concatenate one metric into in-context and cross-context plot columns."""
    in_context_frame = in_context_data.loc[:, ["model", metric]].rename(
        columns={metric: "in_context"}
    )
    in_context_frame["cross_context"] = np.nan

    cross_context_frame = cross_context_data.loc[:, ["model", metric]].rename(
        columns={metric: "cross_context"}
    )
    cross_context_frame["in_context"] = np.nan

    return pd.concat(
        [
            in_context_frame.loc[:, ["model", *PLOT_CONTEXT_COLUMNS]],
            cross_context_frame.loc[:, ["model", *PLOT_CONTEXT_COLUMNS]],
        ],
        ignore_index=True,
    )


def filter_metric_model_order(
    comparison_data: pd.DataFrame, model_order: Sequence[str]
) -> list[str]:
    """Keep models that have at least one finite value in either context."""
    available_models: list[str] = []
    for model in model_order:
        model_values = comparison_data.loc[
            comparison_data["model"] == model, list(PLOT_CONTEXT_COLUMNS)
        ].to_numpy(dtype=float)
        if np.isfinite(model_values).any():
            available_models.append(model)
    return available_models


def metric_limits_for_plot(
    in_context_data: pd.DataFrame,
    cross_context_data: pd.DataFrame,
    metric: str,
) -> dict[str, tuple[float, float]] | None:
    """Return shared y-limits for the in-context and cross-context boxes."""
    metric_limits = compute_metric_limits([in_context_data, cross_context_data], [metric])
    bounds = metric_limits.get(metric)
    if bounds is None:
        return None
    return {plot_metric: bounds for plot_metric in PLOT_CONTEXT_COLUMNS}


def metric_ylabel(metric: str) -> str:
    """Return the y-axis label for one metric."""
    return METRIC_LABELS[metric]


def output_dir_for_comparison(
    in_context_path: Path,
    cross_context_path: Path,
    context_value: str | None = None,
) -> Path:
    """Build the plot output directory for the comparison run."""
    output_dir = (
        in_context_path.parent / f"{in_context_path.stem}_vs_{cross_context_path.stem}_plots"
    )
    if context_value is None:
        return output_dir
    return output_dir / f"context_{context_value.strip().replace('/', '-')}"


def save_metric_plot(
    comparison_data: pd.DataFrame,
    model_order: Sequence[str],
    metric: str,
    output_path: Path,
    metric_limits: dict[str, tuple[float, float]] | None,
) -> bool:
    """Save one metric comparison plot."""
    return save_metric_group_boxplot(
        data=comparison_data,
        model_order=model_order,
        metrics=PLOT_CONTEXT_COLUMNS,
        output_path=output_path,
        title=f"{METRIC_LABELS[metric]}",
        ylabel=metric_axis_label(metric, metric_ylabel(metric)),
        metric_labels=CONTEXT_LABELS,
        metric_base_colors=CONTEXT_COLORS,
        cluster_gap=1.2,
        fig_size=(8.0, 5.8),
        metric_limits=metric_limits,
        dpi=300,
        no_model_message=None,
        x_tick_label_rotation=MODEL_TICK_LABEL_ROTATION,
        x_tick_label_ha=MODEL_TICK_LABEL_ALIGNMENT,
        value_transform=lambda _plot_metric, values: transform_metric_values(values, metric),
    )


def main() -> None:
    """Generate all in-context vs cross-context comparison plots."""
    args = parse_args()

    in_context_path = resolve_results_path(args.in_context_results)
    cross_context_path = resolve_results_path(args.cross_context_results)
    print(f"Using in-context results: {in_context_path}")
    print(f"Using cross-context results: {cross_context_path}")

    in_context_data = filter_context_value(
        prepare_results_data(pd.read_csv(in_context_path)),
        args.context_value,
        label="in-context",
    )
    cross_context_data = filter_context_value(
        prepare_results_data(pd.read_csv(cross_context_path)),
        args.context_value,
        label="cross-context",
    )
    model_order = list(SCLDM_OMEGA_MODEL_ORDER)
    print(f"Models available across inputs: {model_order}")

    output_dir = output_dir_for_comparison(in_context_path, cross_context_path, args.context_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_count = 0
    skipped_metrics: list[str] = []
    for metric in METRICS:
        comparison_data = build_metric_comparison_data(in_context_data, cross_context_data, metric)
        metric_model_order = filter_metric_model_order(comparison_data, model_order)
        if not metric_model_order:
            print(f"Skipping {metric}: no valid values in either file.")
            skipped_metrics.append(metric)
            continue

        output_path = output_dir / f"{metric}.svg"
        if save_metric_plot(
            comparison_data=comparison_data,
            model_order=metric_model_order,
            metric=metric,
            output_path=output_path,
            metric_limits=metric_limits_for_plot(in_context_data, cross_context_data, metric),
        ):
            print(f"Saved: {output_path}")
            generated_count += 1
        else:
            print(f"Skipping {metric}: no valid values to plot.")
            skipped_metrics.append(metric)

    print(f"Generated {generated_count} plot(s) in {output_dir}")
    if skipped_metrics:
        print(f"Skipped metrics: {', '.join(skipped_metrics)}")


if __name__ == "__main__":
    main()
