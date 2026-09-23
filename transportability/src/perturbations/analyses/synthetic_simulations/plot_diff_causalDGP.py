"""Plot causalDGP metrics across change settings for in-context and cross-context runs."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..plot_utils import (
    COMMON_METRIC_LABELS,
    EXPECTED_MODEL_ORDER,
    MODEL_COLORS,
    apply_paper_plot_style,
    coerce_numeric,
    metric_axis_label,
    resolve_expected_model_order,
    transform_metric_values,
    with_vendi_ratio,
)

apply_paper_plot_style()

CHANGE_ORDER: tuple[str, ...] = ("None", "b", "A", "both")
CHANGE_LABELS: dict[str, str] = {
    "None": "None",
    "b": "B",
    "A": "A",
    "both": "Both",
}
PANEL_ORDER: tuple[tuple[str, str], ...] = (
    ("in_context", "In-context"),
    ("cross_context", "Cross-context"),
)
SPLIT_PATH_ARGUMENTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "in_context": (
        ("None", "in_context_no_change", "Path to the in-context no-change results CSV."),
        ("b", "in_context_change_b", "Path to the in-context b-change results CSV."),
        ("A", "in_context_change_A", "Path to the in-context A-change results CSV."),
        ("both", "in_context_change_both", "Path to the in-context both-change results CSV."),
    ),
    "cross_context": (
        ("None", "cross_context_no_change", "Path to the cross-context no-change results CSV."),
        ("b", "cross_context_change_b", "Path to the cross-context b-change results CSV."),
        ("A", "cross_context_change_A", "Path to the cross-context A-change results CSV."),
        ("both", "cross_context_change_both", "Path to the cross-context both-change results CSV."),
    ),
}
PLOT_FIG_SIZE_WITH_LEGEND = (13.2, 5.8)
PLOT_FIG_SIZE_WITHOUT_LEGEND = (11.0, 5.8)
LEGEND_FIG_SIZE = (8.8, 1.8)
LEGEND_OUTPUT_NAME = "model_legend.svg"
MODEL_DODGE_STEP = 0.06
MODEL_LINE_WIDTH = 3.2
MODEL_MARKER_SIZE = 7.5
INTERVAL_LINE_WIDTH = 2.2
GRID_ALPHA = 0.25
VENDI_RATIO_COLUMN = "vendi_score_ratio"
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
    VENDI_RATIO_COLUMN,
    "des_recall",
    "des_precision",
    "des_jaccard",
    "pds_l1",
    "pds_l2",
    "pds_cosine",
)
NUMERIC_RESULT_COLUMNS: tuple[str, ...] = (
    *(metric for metric in METRICS if metric != VENDI_RATIO_COLUMN),
    "vendi_score_pred",
    "vendi_score_obs",
)
METRIC_LABELS: dict[str, str] = COMMON_METRIC_LABELS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot causalDGP metric comparisons between in-context and cross-context "
            "results across the four change settings."
        )
    )
    for split_arguments in SPLIT_PATH_ARGUMENTS.values():
        for _change, arg_name, help_text in split_arguments:
            parser.add_argument(
                f"--{arg_name}",
                required=True,
                type=str,
                help=help_text,
            )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where metric plots will be saved.",
    )
    parser.add_argument(
        "--seperate_legends",
        action="store_true",
        help=(
            "Hide legends inside each metric plot and save one separate legend-only "
            "figure in the output directory."
        ),
    )
    return parser.parse_args(argv)


def model_x_offsets(
    model_order: Sequence[str],
    *,
    step: float = MODEL_DODGE_STEP,
) -> dict[str, float]:
    """Return stable x offsets for the models present in one figure."""
    center = (len(model_order) - 1) / 2.0
    return {model: (index - center) * step for index, model in enumerate(model_order)}


def model_legend_handles(model_order: Sequence[str]) -> list[Line2D]:
    """Build legend handles matching the plotted model lines."""
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=MODEL_LINE_WIDTH,
            markersize=MODEL_MARKER_SIZE,
            label=model,
        )
        for model in model_order
    ]


def resolve_results_path(results_path: str) -> Path:
    """Resolve an input results CSV path and validate that it exists."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return path


def resolve_change_paths(
    args: argparse.Namespace,
    split_key: str,
) -> dict[str, Path]:
    """Return validated change-result paths for one split type."""
    return {
        change: resolve_results_path(getattr(args, arg_name))
        for change, arg_name, _help_text in SPLIT_PATH_ARGUMENTS[split_key]
    }


def default_output_dir(all_paths: Sequence[Path]) -> Path:
    """Return the default output directory for the provided input files."""
    common_parent = Path(os.path.commonpath([str(path.parent.absolute()) for path in all_paths]))
    return common_parent / "different_causalDGP_in_context_vs_cross_context_plots"


def prepare_results_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter successful rows, coerce metrics to numeric, and add vendi ratio."""
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
    prepared = with_vendi_ratio(prepared, ratio_column=VENDI_RATIO_COLUMN)
    if VENDI_RATIO_COLUMN not in prepared.columns:
        prepared[VENDI_RATIO_COLUMN] = np.nan
    return prepared


def build_combined_results(change_results: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all change result tables into one annotated dataframe."""
    frames: list[pd.DataFrame] = []
    for change in CHANGE_ORDER:
        frame = change_results[change].copy()
        frame["change"] = change
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["change"] = pd.Categorical(
        combined["change"],
        categories=CHANGE_ORDER,
        ordered=True,
    )
    return combined


def compute_metric_summary(
    combined_results: pd.DataFrame,
    metric: str,
    model_order: Sequence[str],
) -> pd.DataFrame:
    """Aggregate one metric into per-model median and quartile intervals."""
    grouped = (
        combined_results.groupby(["model", "change"], observed=False)[metric]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            count="count",
        )
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [model_order, CHANGE_ORDER],
        names=["model", "change"],
    )
    summary = (
        grouped.set_index(["model", "change"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["model", "change"])
        .reset_index(drop=True)
    )
    summary["change"] = pd.Categorical(
        summary["change"],
        categories=CHANGE_ORDER,
        ordered=True,
    )

    single_observation = (
        summary["count"].eq(1)
        & summary["median"].notna()
        & summary["q25"].notna()
        & summary["q75"].notna()
    )
    summary.loc[single_observation, "q25"] = summary.loc[single_observation, "median"]
    summary.loc[single_observation, "q75"] = summary.loc[single_observation, "median"]
    return summary


def transform_summary_for_plot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Transform summary statistics into plot space."""
    plot_summary = summary.copy()
    for column in ("median", "q25", "q75"):
        plot_summary[column] = transform_metric_values(
            plot_summary[column].to_numpy(dtype=float),
            metric,
        )
    return plot_summary


def metric_y_limits(*summaries: pd.DataFrame) -> tuple[float, float] | None:
    """Compute shared y limits across one or more transformed summaries."""
    lower_values: list[np.ndarray] = []
    upper_values: list[np.ndarray] = []
    for summary in summaries:
        lower = summary["q25"].to_numpy(dtype=float)
        upper = summary["q75"].to_numpy(dtype=float)
        finite_lower = lower[np.isfinite(lower)]
        finite_upper = upper[np.isfinite(upper)]
        if finite_lower.size:
            lower_values.append(finite_lower)
        if finite_upper.size:
            upper_values.append(finite_upper)

    if not lower_values or not upper_values:
        return None

    y_min = float(np.min(np.concatenate(lower_values)))
    y_max = float(np.max(np.concatenate(upper_values)))
    if np.isclose(y_min, y_max):
        padding = 0.05 * max(abs(y_min), 1.0)
        return y_min - padding, y_max + padding

    padding = 0.05 * (y_max - y_min)
    return y_min - padding, y_max + padding


def draw_change_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    panel_title: str,
    model_order: Sequence[str],
    offsets: dict[str, float],
) -> list[str]:
    """Draw one change panel and return the models that were plotted."""
    x_positions = np.arange(len(CHANGE_ORDER), dtype=float)
    plotted_models: list[str] = []

    for model in model_order:
        model_summary = summary.loc[summary["model"] == model].sort_values("change")
        median_values = model_summary["median"].to_numpy(dtype=float)
        q25_values = model_summary["q25"].to_numpy(dtype=float)
        q75_values = model_summary["q75"].to_numpy(dtype=float)
        valid = np.isfinite(median_values) & np.isfinite(q25_values) & np.isfinite(q75_values)
        if not valid.any():
            continue

        x_valid = x_positions[valid] + offsets[model]
        ax.vlines(
            x_valid,
            q25_values[valid],
            q75_values[valid],
            colors=MODEL_COLORS[model],
            linewidth=INTERVAL_LINE_WIDTH,
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            x_valid,
            median_values[valid],
            color=MODEL_COLORS[model],
            marker="o",
            markersize=MODEL_MARKER_SIZE,
            linewidth=MODEL_LINE_WIDTH,
            zorder=3,
        )
        plotted_models.append(model)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([CHANGE_LABELS[change] for change in CHANGE_ORDER])
    ax.set_xlim(-0.5, len(CHANGE_ORDER) - 0.5)
    ax.set_xlabel("Change")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    return plotted_models


def save_metric_plot(
    in_context_summary: pd.DataFrame,
    cross_context_summary: pd.DataFrame,
    *,
    metric: str,
    model_order: Sequence[str],
    output_path: Path,
    seperate_legends: bool,
) -> bool:
    """Save one two-panel change comparison plot."""
    plot_in_context = transform_summary_for_plot(in_context_summary, metric)
    plot_cross_context = transform_summary_for_plot(cross_context_summary, metric)

    fig_size = PLOT_FIG_SIZE_WITHOUT_LEGEND if seperate_legends else PLOT_FIG_SIZE_WITH_LEGEND
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=True)
    offsets = model_x_offsets(model_order)

    figure_plotted_models: list[str] = []
    for axis, (_panel_key, panel_title), summary in zip(
        np.atleast_1d(axes),
        PANEL_ORDER,
        (plot_in_context, plot_cross_context),
        strict=False,
    ):
        plotted_models = draw_change_panel(
            axis,
            summary,
            panel_title=panel_title,
            model_order=model_order,
            offsets=offsets,
        )
        for model in plotted_models:
            if model not in figure_plotted_models:
                figure_plotted_models.append(model)

    if not figure_plotted_models:
        plt.close(fig)
        return False

    axes_array = np.atleast_1d(axes)
    axes_array[0].set_ylabel(metric_axis_label(metric, METRIC_LABELS[metric]))
    axes_array[1].set_ylabel("")

    y_limits = metric_y_limits(plot_in_context, plot_cross_context)
    if y_limits is not None:
        for axis in axes_array:
            axis.set_ylim(*y_limits)

    if not seperate_legends:
        fig.legend(
            handles=model_legend_handles(figure_plotted_models),
            title="Model",
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),
        )
        fig.subplots_adjust(left=0.08, bottom=0.16, top=0.90, right=0.84, wspace=0.12)
    else:
        fig.subplots_adjust(left=0.08, bottom=0.16, top=0.90, right=0.98, wspace=0.12)

    fig.savefig(output_path, bbox_inches=None)
    plt.close(fig)
    return True


def save_separate_legend(output_path: Path, model_order: Sequence[str]) -> None:
    """Save a standalone figure containing only the model legend."""
    fig, ax = plt.subplots(figsize=LEGEND_FIG_SIZE)
    ax.axis("off")
    ax.legend(
        handles=model_legend_handles(model_order),
        title="Model",
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.2,
        handletextpad=0.6,
    )
    fig.savefig(output_path, bbox_inches=None)
    plt.close(fig)


def main() -> None:
    """Generate causalDGP in-context vs cross-context metric plots."""
    args = parse_args()

    split_change_paths = {
        split_key: resolve_change_paths(args, split_key) for split_key, _panel_title in PANEL_ORDER
    }
    for split_key, change_paths in split_change_paths.items():
        for change, path in change_paths.items():
            print(f"Using {split_key} {change} results: {path}")

    split_change_results = {
        split_key: {
            change: prepare_results_data(pd.read_csv(path)) for change, path in change_paths.items()
        }
        for split_key, change_paths in split_change_paths.items()
    }
    model_order = resolve_expected_model_order(
        *[
            data
            for change_results in split_change_results.values()
            for data in change_results.values()
        ],
        expected_order=EXPECTED_MODEL_ORDER,
    )
    print(f"Models available across inputs: {model_order}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(
            [path for change_paths in split_change_paths.values() for path in change_paths.values()]
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_results_by_split = {
        split_key: build_combined_results(change_results)
        for split_key, change_results in split_change_results.items()
    }

    if args.seperate_legends:
        legend_output_path = output_dir / LEGEND_OUTPUT_NAME
        save_separate_legend(legend_output_path, model_order)
        print(f"Saved separate legend: {legend_output_path}")

    generated_count = 0
    skipped_metrics: list[str] = []
    for metric in METRICS:
        output_path = output_dir / f"{metric}.svg"
        if save_metric_plot(
            compute_metric_summary(
                combined_results_by_split["in_context"],
                metric,
                model_order,
            ),
            compute_metric_summary(
                combined_results_by_split["cross_context"],
                metric,
                model_order,
            ),
            metric=metric,
            model_order=model_order,
            output_path=output_path,
            seperate_legends=args.seperate_legends,
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
