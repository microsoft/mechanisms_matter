"""Plot observed DEG percentage against evaluation metrics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from ..plot_utils import (
    COMMON_METRIC_LABELS,
    add_pearson_correlation_annotation,
    apply_paper_plot_style,
    coerce_numeric,
    compute_axis_limits,
    compute_pearson_correlation,
    compute_trend_line,
    get_available_model_order,
    get_model_colors,
    metric_axis_label,
    transform_metric_values,
)

apply_paper_plot_style(
    {
        "figure.figsize": (7, 6),
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
    }
)

DEFAULT_WINDOW = 20
DEG_PERCENTAGE_COLUMN = "deg_percentage"
EXCLUDED_MODELS: tuple[str, ...] = ("Context-Average", "Context-linearPCA")
METRIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("pearson", "pearson_degs"),
    ("mae", "mae_degs"),
    ("mse", "mse_degs"),
    ("r2", "r2_degs"),
    ("mmd_distance", "parametric_distance", "fid_distance"),
)
DISTANCE_METRICS_GROUP: tuple[str, ...] = (
    "mmd_distance",
    "parametric_distance",
    "fid_distance",
)
ALIGNED_Y_RANGE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("pearson", "pearson_degs"),
    ("r2", "r2_degs"),
)
GROUP_OUTPUT_STEMS: dict[tuple[str, ...], str] = {
    ("pearson", "pearson_degs"): "pearson",
    ("mae", "mae_degs"): "mae",
    ("mse", "mse_degs"): "mse",
    ("r2", "r2_degs"): "r2",
    DISTANCE_METRICS_GROUP: "distance_metrics",
}
NUMERIC_RESULT_COLUMNS: tuple[str, ...] = (
    "n_obs_degs",
    "n_genes",
    *(metric for group in METRIC_GROUPS for metric in group),
)


def prepare_results_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter successful rows and add the observed DEG percentage."""
    prepared = data.copy()
    prepared["model"] = prepared["model"].astype("string").str.strip()
    prepared = prepared[prepared["status"].astype(str).str.lower() == "success"].copy()
    prepared = prepared[~prepared["model"].isin(EXCLUDED_MODELS)].copy()
    coerce_numeric(prepared, NUMERIC_RESULT_COLUMNS)
    prepared[DEG_PERCENTAGE_COLUMN] = prepared["n_obs_degs"].div(
        prepared["n_genes"].where(prepared["n_genes"].ne(0))
    )
    return prepared


def extract_plot_points(
    data: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite x values plus raw and transformed metric values."""
    x_values = pd.to_numeric(data[DEG_PERCENTAGE_COLUMN], errors="coerce").to_numpy(dtype=float)
    y_raw_values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(x_values) & np.isfinite(y_raw_values)

    x_values = x_values[valid_mask]
    y_raw_values = y_raw_values[valid_mask]
    y_plot_values = transform_metric_values(y_raw_values, metric)

    finite_plot_mask = np.isfinite(y_plot_values)
    return (
        x_values[finite_plot_mask],
        y_raw_values[finite_plot_mask],
        y_plot_values[finite_plot_mask],
    )


def draw_metric_panel(
    axis,
    data: pd.DataFrame,
    *,
    metric: str,
    model_order: Sequence[str],
    model_colors: dict[str, str],
    window: int,
    plot_scatters: bool,
    use_loess: bool,
) -> tuple[list[str], np.ndarray]:
    """Draw one metric panel and return rendered models plus plotted y-values."""
    x_all, y_raw_all, _ = extract_plot_points(data, metric)
    if x_all.size == 0:
        axis.set_visible(False)
        return [], np.asarray([], dtype=float)

    plotted_models: list[str] = []
    displayed_y_values: list[np.ndarray] = []
    for model_name in model_order:
        model_data = data[data["model"] == model_name]
        x_values, y_raw_values, y_plot_values = extract_plot_points(model_data, metric)
        if x_values.size == 0:
            continue

        color = model_colors[model_name]
        drew_model = False
        if plot_scatters:
            axis.scatter(
                x_values,
                y_plot_values,
                alpha=0.28,
                s=20,
                color=color,
                rasterized=True,
            )
            drew_model = True
            displayed_y_values.append(y_plot_values)

        trend_line = compute_trend_line(
            x_values,
            y_raw_values,
            window=window,
            use_loess=use_loess,
        )

        if trend_line is not None:
            x_average, y_average = trend_line
            y_average_plot = transform_metric_values(y_average, metric)
            finite_average_mask = np.isfinite(y_average_plot)
            if finite_average_mask.any():
                axis.plot(
                    x_average[finite_average_mask],
                    y_average_plot[finite_average_mask],
                    color=color,
                    linestyle="--",
                    linewidth=3.0,
                )
                drew_model = True
                displayed_y_values.append(y_average_plot[finite_average_mask])

        if drew_model:
            plotted_models.append(model_name)

    if not plotted_models:
        axis.set_visible(False)
        return [], np.asarray([], dtype=float)

    x_limits = compute_axis_limits(x_all, bounds=(0.0, 1.0))
    if x_limits is not None:
        axis.set_xlim(*x_limits)

    metric_label = COMMON_METRIC_LABELS.get(metric, metric)
    axis.set_xlabel("Observed DEG Percentage")
    axis.set_ylabel(metric_axis_label(metric, metric_label))
    corr, p_value = compute_pearson_correlation(x_all, y_raw_all)
    add_pearson_correlation_annotation(axis, corr, p_value)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    sns.despine(ax=axis)
    return plotted_models, np.concatenate(displayed_y_values)


def plot_metric_group_vs_deg_percentage(
    data: pd.DataFrame,
    *,
    metrics: Sequence[str],
    output_path: Path,
    window: int,
    plot_scatters: bool,
    use_loess: bool,
) -> bool:
    """Save one multi-panel DEG-percentage plot for related metrics."""
    missing_metrics = [metric for metric in metrics if metric not in data.columns]
    if missing_metrics:
        print(f"Skipping {tuple(metrics)}: missing columns {missing_metrics}.")
        return False

    model_order = get_available_model_order(data)
    if not model_order:
        print(f"Skipping {tuple(metrics)}: no models available after filtering.")
        return False
    model_colors = get_model_colors(model_order)

    figure_width = 6.4 * len(metrics)
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(figure_width, 5.8),
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    figure_plotted_models: list[str] = []
    legend_axis = None
    figure_y_values: list[np.ndarray] = []

    for axis, metric in zip(axes_array, metrics, strict=False):
        plotted_models, panel_y_values = draw_metric_panel(
            axis,
            data,
            metric=metric,
            model_order=model_order,
            model_colors=model_colors,
            window=window,
            plot_scatters=plot_scatters,
            use_loess=use_loess,
        )
        if plotted_models:
            legend_axis = axis
        if panel_y_values.size:
            figure_y_values.append(panel_y_values)
        for model_name in plotted_models:
            if model_name not in figure_plotted_models:
                figure_plotted_models.append(model_name)

    if not figure_plotted_models:
        plt.close(figure)
        print(f"Skipping {tuple(metrics)}: no drawable points available.")
        return False

    if tuple(metrics) in ALIGNED_Y_RANGE_GROUPS:
        shared_y_limits = compute_axis_limits(np.concatenate(figure_y_values))
        if shared_y_limits is not None:
            for axis in axes_array:
                if axis.get_visible():
                    axis.set_ylim(*shared_y_limits)

    if legend_axis is not None:
        legend_axis.legend(
            handles=[
                Patch(
                    facecolor=model_colors[model_name],
                    edgecolor="none",
                    label=model_name,
                )
                for model_name in figure_plotted_models
            ],
            title="Model",
            loc="best",
        )

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    print(f"Generated {output_path}")
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot observed DEG percentage against evaluation metrics.",
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to a DEG-vs-metrics CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory for the generated plots.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Moving-average window size per model (default: {DEFAULT_WINDOW}).",
    )
    parser.add_argument(
        "--plot_scatters",
        action="store_true",
        help="Include scatter points.",
    )
    parser.add_argument(
        "--use_loess",
        action="store_true",
        help="Use LOESS for the trend line.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate all DEG-percentage distortion plots for one results table."""
    args = parse_args(argv)
    results_path = Path(args.results)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else results_path.parent / f"{results_path.stem}_metrics_distortion_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using results: {results_path}")
    print(f"Saving plots to: {output_dir}")
    data = prepare_results_data(pd.read_csv(results_path))
    model_order = get_available_model_order(data)
    print(
        f"Prepared {len(data)} successful rows "
        f"(models: {model_order if model_order else 'not available'})."
    )

    generated_count = 0
    for metrics in METRIC_GROUPS:
        file_stem = GROUP_OUTPUT_STEMS[tuple(metrics)]
        if plot_metric_group_vs_deg_percentage(
            data=data,
            metrics=metrics,
            output_path=output_dir / f"{file_stem}_group_vs_deg_percentage.svg",
            window=args.window,
            plot_scatters=args.plot_scatters,
            use_loess=args.use_loess,
        ):
            generated_count += 1

    print(f"Finished generating {generated_count} plots from {results_path}.")


if __name__ == "__main__":
    main()
