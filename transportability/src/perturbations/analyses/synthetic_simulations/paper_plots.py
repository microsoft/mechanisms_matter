"""Paper-ready plotting entrypoint for synthetic perturbation simulation results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from ..plot_utils import (
    COMMON_METRIC_BASE_COLORS,
    COMMON_METRIC_LABELS,
    MODEL_COLORS,
    MODEL_TICK_LABEL_ALIGNMENT,
    MODEL_TICK_LABEL_ROTATION,
    add_pearson_correlation_annotation,
    apply_paper_plot_style,
    coerce_numeric,
    compute_axis_limits,
    compute_pearson_correlation,
    compute_trend_line,
    get_available_model_order,
    metric_axis_label,
    save_metric_group_boxplot,
    transform_metric_bounds,
    transform_metric_values,
    with_vendi_ratio,
)

# Set up a clean, professional style similar to science plots
apply_paper_plot_style(
    {
        "figure.figsize": (7, 6),
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
    }
)

PARAMETER_PLOT_SPECS = (
    {
        "x_column": "B",
        "file_suffix": "control_bias",
        "title_fragment": r"$\beta$",
        "x_label": "Control Bias (β)",
        "log_x": False,
    },
    {
        "x_column": "N0",
        "file_suffix": "n0",
        "title_fragment": r"$n_0$",
        "x_label": "Number of Control Cells ($n_0$)",
        "log_x": True,
    },
    {
        "x_column": "P",
        "file_suffix": "perturbations",
        "title_fragment": "Number of Perturbations",
        "x_label": "Number of Perturbations ($P$)",
        "log_x": False,
    },
    {
        "x_column": "sparsity",
        "file_suffix": "sparsity",
        "title_fragment": "Sparsity",
        "x_label": "Sparsity",
        "log_x": False,
    },
    {
        "x_column": "systematic_variation",
        "file_suffix": "systematic_variation",
        "title_fragment": "Systematic Variation",
        "x_label": "Systematic Variation",
        "log_x": False,
    },
    {
        "x_column": "intra_corr",
        "file_suffix": "intra_corr",
        "title_fragment": "Intra-data Correlation",
        "x_label": "Intra-data Correlation",
        "log_x": False,
    },
)
CONTEXT_BASELINE_MODELS = ("Context-Average", "Context-linearPCA")
DEG_METRIC_BASES = ("pearson", "mae", "mse", "r2")
DEG_METRIC_SUFFIXES = ("degs", "true_degs")
DES_BOX_METRICS = ("des_recall", "des_precision", "des_jaccard")
SINGLE_BOX_METRICS = (
    "parametric_distance",
    "mmd_distance",
    "fid_distance",
    *DES_BOX_METRICS,
)
PDS_BOX_METRICS = ("pds_l1", "pds_l2", "pds_cosine")
DEG_VARIANT_METRICS = tuple(
    f"{metric}_{suffix}" for metric in DEG_METRIC_BASES for suffix in DEG_METRIC_SUFFIXES
)
NUMERIC_RESULT_COLUMNS = (
    *DEG_METRIC_BASES,
    *DEG_VARIANT_METRICS,
    *SINGLE_BOX_METRICS,
    *PDS_BOX_METRICS,
    "vendi_ratio",
    "vendi_score_pred",
    "vendi_score_obs",
)
METRIC_LABELS = COMMON_METRIC_LABELS
METRIC_BASE_COLORS = COMMON_METRIC_BASE_COLORS
EXTRA_METRIC_LABELS = {
    "vendi_score": "Vendi Score",
    "intra_corr": "Intra-data Correlation",
}
PEARSON_ROBUST_QUANTILES = (0.01, 0.99)
PEARSON_ROBUST_MIN_POINTS = 20


def selected_deg_metric_suffix(use_true_deg: bool) -> str:
    """Return the metric suffix for DEG-specific plots."""
    return "true_degs" if use_true_deg else "degs"


def selected_deg_metric_column(metric: str, use_true_deg: bool) -> str:
    """Return the DEG-specific metric column for the requested variant."""
    return f"{metric}_{selected_deg_metric_suffix(use_true_deg)}"


def selected_deg_metric_label(use_true_deg: bool) -> str:
    """Return the display label for the selected DEG metric variant."""
    return "Affected Genes" if use_true_deg else "DEGs"


def selected_pair_metric_groups(use_true_deg: bool) -> tuple[tuple[str, str], ...]:
    """Return paired metric groups for the selected DEG metric variant."""
    return tuple(
        (metric, selected_deg_metric_column(metric, use_true_deg)) for metric in DEG_METRIC_BASES
    )


def metric_label(metric: str) -> str:
    """Return the shared display label for one metric."""
    return EXTRA_METRIC_LABELS.get(metric, METRIC_LABELS.get(metric, metric))


def robust_plot_y_limits(
    metric: str,
    scatter_values: Sequence[np.ndarray],
    trend_values: Sequence[np.ndarray],
) -> tuple[float, float] | None:
    """Return robust y-axis limits for Pearson-family plots."""
    if not metric.startswith("pearson"):
        return None

    plotted_values = [
        values
        for values in (*scatter_values, *trend_values)
        if np.asarray(values, dtype=float).size > 0
    ]
    if not plotted_values:
        return None

    all_values = np.concatenate(plotted_values)
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size < PEARSON_ROBUST_MIN_POINTS:
        return None

    lower, upper = np.quantile(finite_values, PEARSON_ROBUST_QUANTILES)
    nonempty_trend_values = [
        values for values in trend_values if np.asarray(values, dtype=float).size > 0
    ]
    if nonempty_trend_values:
        trend_all = np.concatenate(nonempty_trend_values)
        finite_trend = trend_all[np.isfinite(trend_all)]
        if finite_trend.size > 0:
            lower = min(lower, float(np.min(finite_trend)))
            upper = max(upper, float(np.max(finite_trend)))

    return compute_axis_limits(np.asarray([lower, upper]), bounds=(-1.0, 1.0))


def prepare_xy(x, y):
    """Convert x/y to numeric arrays and drop non-finite values."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def select_statistics_subset(data):
    """
    Keep one model's rows for statistics plots to avoid repeated trial-level values.

    Returns (subset_dataframe, selected_model_name_or_none).
    """
    available_models = get_available_model_order(data)
    if not available_models:
        return data.copy(), None
    selected_model = available_models[0]
    subset = data[data["model"] == selected_model].copy()
    return subset, selected_model


def plot_metric_vs_parameter(
    data,
    save_path,
    x_column,
    y_column,
    title,
    x_label,
    y_label=None,
    window=50,
    y_lim=None,
    log_x=False,
    log_x_if_range=False,
    corr_log_x=None,
    log_x_range_threshold=10,
    color_by_model=False,
    use_loess=False,
):
    """Generic scatter + moving-average plot with Pearson correlation annotation."""
    _, ax = plt.subplots(figsize=(10, 6) if color_by_model else (7, 6))
    if y_label is None:
        y_label = metric_label(y_column)
    scatter_plot_values: list[np.ndarray] = []
    trend_plot_values: list[np.ndarray] = []

    y_series = pd.to_numeric(data[y_column], errors="coerce")
    x_series = pd.to_numeric(data[x_column], errors="coerce")
    x, y = prepare_xy(x_series, y_series)

    # Decide whether to use log scale on x-axis
    use_log = log_x
    if log_x_if_range and len(x) > 0:
        min_x = np.min(x)
        max_x = np.max(x)
        if min_x > 0 and (max_x / min_x) > log_x_range_threshold:
            use_log = True

    # Scatter and trendline drawing
    model_order = get_available_model_order(data)
    plotted_models = []
    if color_by_model and model_order:
        for model_name in model_order:
            model_mask = data["model"] == model_name
            x_m = pd.to_numeric(data.loc[model_mask, x_column], errors="coerce")
            y_m = pd.to_numeric(data.loc[model_mask, y_column], errors="coerce")
            x_model, y_model_raw = prepare_xy(x_m, y_m)
            y_model = transform_metric_values(y_model_raw, y_column)
            finite_plot_mask = np.isfinite(y_model)
            x_model = x_model[finite_plot_mask]
            y_model_raw = y_model_raw[finite_plot_mask]
            y_model = y_model[finite_plot_mask]
            if len(x_model) == 0:
                continue

            color = MODEL_COLORS.get(model_name, None)
            ax.scatter(x_model, y_model, alpha=0.28, s=20, color=color, label=model_name)
            scatter_plot_values.append(y_model)

            trend_line = compute_trend_line(
                x_model,
                y_model_raw,
                window=window,
                use_loess=use_loess,
            )
            if trend_line is not None:
                x_ma, y_ma = trend_line
                if len(x_ma) > 0:
                    y_ma_plot = transform_metric_values(y_ma, y_column)
                    finite_ma_mask = np.isfinite(y_ma_plot)
                    y_ma_plot = y_ma_plot[finite_ma_mask]
                    if y_ma_plot.size == 0:
                        continue
                    trend_plot_values.append(y_ma_plot)
                    ax.plot(
                        x_ma[finite_ma_mask],
                        y_ma_plot,
                        color=color,
                        linestyle="--",
                        linewidth=3.0,
                    )
            plotted_models.append(model_name)
    else:
        # Default single-color plotting path
        y_plot = transform_metric_values(y, y_column)
        finite_plot_mask = np.isfinite(y_plot)
        x_plot = x[finite_plot_mask]
        y_plot = y_plot[finite_plot_mask]
        y_raw_plot = y[finite_plot_mask]
        ax.scatter(x_plot, y_plot, alpha=0.3, s=20, color="tab:blue")
        scatter_plot_values.append(y_plot)
        trend_line = compute_trend_line(
            x_plot,
            y_raw_plot,
            window=window,
            use_loess=use_loess,
        )
        if trend_line is not None:
            x_ma, y_ma = trend_line
            if len(x_ma) > 0:
                y_ma_plot = transform_metric_values(y_ma, y_column)
                finite_ma_mask = np.isfinite(y_ma_plot)
                y_ma_plot = y_ma_plot[finite_ma_mask]
                if y_ma_plot.size == 0:
                    trend_line = None
                else:
                    trend_plot_values.append(y_ma_plot)
                    ax.plot(
                        x_ma[finite_ma_mask],
                        y_ma_plot,
                        color="navy",
                        linestyle="--",
                        linewidth=3.0,
                    )

    # Calculate Pearson correlation
    if corr_log_x is None:
        corr_log_x = use_log
    corr, p_value = compute_pearson_correlation(x, y, log_x=corr_log_x)

    # Set axis labels
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(metric_axis_label(y_column, y_label), fontsize=20)

    if y_lim is not None:
        ax.set_ylim(*transform_metric_bounds(y_lim, y_column))
    else:
        robust_y_limits = robust_plot_y_limits(
            y_column,
            scatter_values=scatter_plot_values,
            trend_values=trend_plot_values,
        )
        if robust_y_limits is not None:
            ax.set_ylim(*robust_y_limits)

    # Add correlation text at leftmost limit with lowered y position
    add_pearson_correlation_annotation(ax, corr, p_value)

    # Use log scale for x-axis if requested
    if use_log:
        ax.set_xscale("log")

    if plotted_models:
        legend_handles = [
            Patch(
                facecolor=MODEL_COLORS.get(model_name, "gray"),
                edgecolor="none",
                label=model_name,
            )
            for model_name in plotted_models
        ]
        ax.legend(
            handles=legend_handles,
            title="Model",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

    # Remove top and right spines
    sns.despine()

    # Save figure as PDF
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Generated {save_path}")


def plot_pds_vs_parameter(
    data,
    save_path,
    x_column,
    title,
    x_label,
    window=50,
    pds_metric=None,
    log_x=False,
    log_x_if_range=False,
    color_by_model=True,
    use_loess=False,
):
    """Plot PDS vs a parameter with a moving average trend line."""
    if pds_metric is None:
        raise ValueError("pds_metric must be provided for synthetic paper plots.")

    plot_metric_vs_parameter(
        data=data,
        save_path=save_path,
        x_column=x_column,
        y_column=pds_metric,
        title=title,
        x_label=x_label,
        window=window,
        log_x=log_x,
        log_x_if_range=log_x_if_range,
        color_by_model=color_by_model,
        use_loess=use_loess,
    )


def plot_pearson_delta_vs_parameter(
    data,
    save_path,
    x_column,
    title,
    x_label,
    window=50,
    y_column="pearson",
    y_label=None,
    log_x=False,
    log_x_if_range=False,
    corr_log_x=None,
    color_by_model=True,
    use_loess=False,
):
    """Plot Pearson delta vs a parameter with a moving average trend line."""
    plot_metric_vs_parameter(
        data=data,
        save_path=save_path,
        x_column=x_column,
        y_column=y_column,
        title=title,
        x_label=x_label,
        y_label=y_label,
        window=window,
        log_x=log_x,
        log_x_if_range=log_x_if_range,
        corr_log_x=corr_log_x,
        color_by_model=color_by_model,
        use_loess=use_loess,
    )


def plot_statistics(
    data,
    save_path,
    column="sparsity",
    bins=30,
    x_label=None,
    y_label="Frequency",
    title=None,
    log_x=False,
):
    """
    Plot a histogram for any numeric column in the results table.

    :param data: Input dataframe
    :param save_path: Path to save the figure
    :param column: Column name to plot
    :param bins: Number of histogram bins
    :param x_label: Optional x-axis label (defaults to title-cased column name)
    :param y_label: y-axis label
    :param title: Optional figure title (defaults to 'Histogram of <x_label>')
    :param log_x: Whether to use log scale on x-axis
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")

    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Column '{column}' has no numeric values to plot.")

    if log_x:
        positive_values = values[values > 0]
        dropped = len(values) - len(positive_values)
        if positive_values.empty:
            raise ValueError(f"Column '{column}' has no positive values for log x-axis.")
        if dropped > 0:
            print(
                f"Dropped {dropped} non-positive values from '{column}' for log x-axis histogram."
            )
        values = positive_values

    hist_bins = bins
    if log_x and isinstance(bins, int):
        vmin = values.min()
        vmax = values.max()
        if np.isclose(vmin, vmax):
            hist_bins = bins
        else:
            hist_bins = np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)

    if x_label is None:
        x_label = column.replace("_", " ").title()
    if title is None:
        title = f"Histogram of {x_label}"

    _, ax = plt.subplots(figsize=(7, 6))
    ax.hist(values, bins=hist_bins, color="skyblue", edgecolor="black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Generated {save_path}")


def plot_metric_family_vs_parameters(
    data,
    plot_dir,
    metric_column,
    file_stem,
    title_prefix,
    y_label=None,
    window=50,
    y_lim=None,
    color_by_model=True,
    use_loess=False,
):
    """Generate a standard six-panel metric family across the sweep parameters."""
    plot_dir = Path(plot_dir)
    if metric_column not in data.columns:
        print(f"Skipping metric family '{metric_column}': column not found.")
        return
    if y_label is None:
        y_label = metric_label(metric_column)

    metric_values = pd.to_numeric(data[metric_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(metric_values).any():
        print(f"Skipping metric family '{metric_column}': no valid values.")
        return

    for spec in PARAMETER_PLOT_SPECS:
        plot_metric_vs_parameter(
            data=data,
            save_path=plot_dir / f"{file_stem}_vs_{spec['file_suffix']}.svg",
            x_column=spec["x_column"],
            y_column=metric_column,
            title=f"{title_prefix} by {spec['title_fragment']}",
            x_label=spec["x_label"],
            y_label=y_label,
            window=window,
            y_lim=y_lim,
            log_x=bool(spec.get("log_x", False)),
            color_by_model=color_by_model,
            use_loess=use_loess,
        )


def prepare_results_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter successful runs and coerce plotted metrics to numeric."""
    prepared = data.copy()
    prepared["model"] = prepared["model"].astype("string").str.strip()
    prepared = prepared[prepared["status"].astype(str).str.lower() == "success"].copy()
    coerce_numeric(prepared, NUMERIC_RESULT_COLUMNS)
    return with_vendi_ratio(prepared)


def available_model_order_for_metrics(
    data: pd.DataFrame,
    model_order: Sequence[str],
    metrics: Sequence[str],
) -> list[str]:
    """Keep only models with at least one finite value in the requested metrics."""
    available_models: list[str] = []
    for model_name in model_order:
        model_data = data.loc[data["model"] == model_name]
        if model_data.empty:
            continue

        has_values = False
        for metric in metrics:
            if metric not in model_data.columns:
                continue
            raw_values = pd.to_numeric(model_data[metric], errors="coerce").to_numpy(dtype=float)
            plot_values = transform_metric_values(raw_values, metric)
            if np.isfinite(plot_values).any():
                has_values = True
                break

        if has_values:
            available_models.append(model_name)

    return available_models


def save_boxplot(
    data: pd.DataFrame,
    model_order: Sequence[str],
    metrics: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
    cluster_gap: float = 1.2,
) -> bool:
    """Save one grouped boxplot with the shared synthetic-plot styling."""
    available_models = available_model_order_for_metrics(data, model_order, metrics)
    if not available_models:
        return False

    return save_metric_group_boxplot(
        data=data,
        model_order=available_models,
        metrics=metrics,
        output_path=output_path,
        title=title,
        ylabel=ylabel,
        metric_labels=METRIC_LABELS,
        metric_base_colors=METRIC_BASE_COLORS,
        cluster_gap=cluster_gap,
        fig_size=(8.0, 5.8),
        x_tick_label_rotation=MODEL_TICK_LABEL_ROTATION,
        x_tick_label_ha=MODEL_TICK_LABEL_ALIGNMENT,
    )


def parse_args(argv: Sequence[str] | None = None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Pearson delta and PDS plots from simulation results."
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the CSV file with simulation results",
    )
    parser.add_argument(
        "--true_deg",
        action="store_true",
        help="Use *_true_degs metrics instead of *_degs for DEG-specific plots.",
    )
    parser.add_argument(
        "--remove_context_baselines",
        action="store_true",
        help="Remove Context-Average and Context-linearPCA from all plots.",
    )
    parser.add_argument(
        "--use_loess",
        action="store_true",
        help="Use LOESS instead of moving averages for trend lines.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Generate the complete synthetic-simulation figure set from result CSV files."""
    args = parse_args()
    results_path = Path(args.results)
    print(f"Using results: {results_path}")
    data = prepare_results_data(pd.read_csv(results_path))
    selected_deg_label = selected_deg_metric_label(args.true_deg)
    selected_deg_suffix = selected_deg_metric_suffix(args.true_deg)
    selected_pearson_deg_metric = selected_deg_metric_column("pearson", args.true_deg)
    selected_r2_deg_metric = selected_deg_metric_column("r2", args.true_deg)

    performance_data = data.copy()
    if args.remove_context_baselines:
        performance_data = performance_data[
            ~performance_data["model"].isin(CONTEXT_BASELINE_MODELS)
        ].copy()
    model_order = get_available_model_order(performance_data)
    if model_order:
        performance_data = performance_data[performance_data["model"].isin(model_order)].copy()
    print(
        f"Performance plotting rows: {len(performance_data)} "
        f"(models: {model_order if model_order else 'not available'})"
    )

    statistics_data, statistics_model = select_statistics_subset(performance_data)
    if statistics_model is not None:
        print(
            f"Statistics plots use model '{statistics_model}' only ({len(statistics_data)} rows)."
        )
    else:
        print(f"Statistics plots use all available rows ({len(statistics_data)} rows).")

    window = 20
    use_loess = args.use_loess

    plot_dir = results_path.parent / f"{results_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for left_metric, right_metric in selected_pair_metric_groups(args.true_deg):
        output_path = plot_dir / f"{left_metric}_vs_{right_metric}_boxplot.svg"
        title = f"{METRIC_LABELS[left_metric]} vs {METRIC_LABELS[right_metric]}"
        if save_boxplot(
            data=performance_data,
            model_order=model_order,
            metrics=[left_metric, right_metric],
            output_path=output_path,
            title=title,
            ylabel="Score",
        ):
            print(f"Generated {output_path}")

    for metric in SINGLE_BOX_METRICS:
        output_path = plot_dir / f"{metric}_boxplot.svg"
        if save_boxplot(
            data=performance_data,
            model_order=model_order,
            metrics=[metric],
            output_path=output_path,
            title=f"{METRIC_LABELS[metric]} by Model",
            ylabel="Distance" if metric.endswith("_distance") else "Score",
        ):
            print(f"Generated {output_path}")

    vendi_output_path = plot_dir / "vendi_ratio_boxplot.svg"
    if save_boxplot(
        data=performance_data,
        model_order=model_order,
        metrics=["vendi_ratio"],
        output_path=vendi_output_path,
        title="Vendi Score Ratio by Model",
        ylabel="Ratio",
    ):
        print(f"Generated {vendi_output_path}")

    pds_output_path = plot_dir / "pds_metrics_boxplots.svg"
    if save_boxplot(
        data=performance_data,
        model_order=model_order,
        metrics=PDS_BOX_METRICS,
        output_path=pds_output_path,
        title="PDS Metrics by Model",
        ylabel="Score",
    ):
        print(f"Generated {pds_output_path}")

    # Plot Pearson delta vs control bias (β)
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / "pearson_delta_vs_control_bias.svg"),
        x_column="B",
        title=r"$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{β}$  $\mathbf{(Simulation)}$",
        x_label="Control Bias (β)",
        window=window,
        corr_log_x=False,
        use_loess=use_loess,
    )
    # Plot Pearson delta vs n0
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / "pearson_delta_vs_n0.svg"),
        x_column="N0",
        title=r"$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{n_0}$ $\mathbf{(Simulation)}$",
        x_label="Number of Control Cells ($n_0$)",
        window=window,
        log_x=True,
        corr_log_x=True,
        use_loess=use_loess,
    )
    # Plot Pearson delta vs number of perturbations
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / "pearson_delta_vs_perturbations.svg"),
        x_column="P",
        title=r"$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{k}$ $\mathbf{(Simulation)}$",
        x_label="Number of Perturbations ($P$)",
        window=window,
        log_x_if_range=True,
        corr_log_x=True,
        use_loess=use_loess,
    )
    # Plot Pearson delta vs sparsity
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / "pearson_delta_vs_sparsity.svg"),
        x_column="sparsity",
        title=r"$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{Sparsity}$  $\mathbf{(Simulation)}$",
        x_label="Sparsity",
        window=window,
        log_x=False,
        corr_log_x=False,
        use_loess=use_loess,
    )
    # Plot Pearson delta vs systematic variation
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / "pearson_delta_vs_systematic_variation.svg"),
        x_column="systematic_variation",
        title=r"$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{Systematic}$ $\mathbf{Variation}$  $\mathbf{(Simulation)}$",
        x_label="Systematic Variation",
        window=window,
        log_x=False,
        corr_log_x=False,
        use_loess=use_loess,
    )
    # Plot Pearson delta vs intra-data correlation
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / "pearson_delta_vs_intra_corr.svg"),
        x_column="intra_corr",
        title=r"$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{Intra}$-$\mathbf{data}$ $\mathbf{Correlation}$  $\mathbf{(Simulation)}$",
        x_label="Intra-data Correlation",
        window=window,
        log_x=False,
        corr_log_x=False,
        use_loess=use_loess,
    )
    # Plot Pearson delta for the selected DEG metric vs control bias (β)
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / f"pearson_delta_{selected_deg_suffix}_vs_control_bias.svg"),
        x_column="B",
        y_column=selected_pearson_deg_metric,
        title=f"Pearson(Δ) ({selected_deg_label}) by β (Simulation)",
        x_label="Control Bias (β)",
        window=window,
        corr_log_x=False,
        use_loess=use_loess,
    )
    # Plot Pearson delta for the selected DEG metric vs n0
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / f"pearson_delta_{selected_deg_suffix}_vs_n0.svg"),
        x_column="N0",
        y_column=selected_pearson_deg_metric,
        title=f"Pearson(Δ) ({selected_deg_label}) by n_0 (Simulation)",
        x_label="Number of Control Cells ($n_0$)",
        window=window,
        log_x=True,
        corr_log_x=True,
        use_loess=use_loess,
    )
    # Plot Pearson delta for the selected DEG metric vs number of perturbations.
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / f"pearson_delta_{selected_deg_suffix}_vs_perturbations.svg"),
        x_column="P",
        y_column=selected_pearson_deg_metric,
        title=f"Pearson(Δ) ({selected_deg_label}) by k (Simulation)",
        x_label="Number of Perturbations ($P$)",
        window=window,
        log_x_if_range=True,
        corr_log_x=True,
        use_loess=use_loess,
    )
    # Plot Pearson delta for the selected DEG metric vs sparsity.
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / f"pearson_delta_{selected_deg_suffix}_vs_sparsity.svg"),
        x_column="sparsity",
        y_column=selected_pearson_deg_metric,
        title=f"Pearson(Δ) ({selected_deg_label}) by Sparsity (Simulation)",
        x_label="Sparsity",
        window=window,
        log_x=False,
        corr_log_x=False,
        use_loess=use_loess,
    )
    # Plot Pearson delta for the selected DEG metric vs systematic variation.
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / f"pearson_delta_{selected_deg_suffix}_vs_systematic_variation.svg"),
        x_column="systematic_variation",
        y_column=selected_pearson_deg_metric,
        title=f"Pearson(Δ) ({selected_deg_label}) by Systematic Variation (Simulation)",
        x_label="Systematic Variation",
        window=window,
        log_x=False,
        corr_log_x=False,
        use_loess=use_loess,
    )
    # Plot Pearson delta for the selected DEG metric vs intra-data correlation.
    plot_pearson_delta_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / f"pearson_delta_{selected_deg_suffix}_vs_intra_corr.svg"),
        x_column="intra_corr",
        y_column=selected_pearson_deg_metric,
        title=f"Pearson(Δ) ({selected_deg_label}) by Intra-data Correlation (Simulation)",
        x_label="Intra-data Correlation",
        window=window,
        log_x=False,
        corr_log_x=False,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="vendi_ratio",
        file_stem="vendi_ratio",
        title_prefix="Vendi Score Ratio",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="parametric_distance",
        file_stem="parametric_distance",
        title_prefix="Parametric Distance",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="mmd_distance",
        file_stem="mmd_distance",
        title_prefix="MMD",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="fid_distance",
        file_stem="fid_distance",
        title_prefix="FD Distance",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="des_recall",
        file_stem="des_recall",
        title_prefix="DES (Recall)",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="des_precision",
        file_stem="des_precision",
        title_prefix="DES (Precision)",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="des_jaccard",
        file_stem="des_jaccard",
        title_prefix="DES (Jaccard)",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column="r2",
        file_stem="r2",
        title_prefix=r"$R^2$",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_family_vs_parameters(
        data=performance_data,
        plot_dir=plot_dir,
        metric_column=selected_r2_deg_metric,
        file_stem=selected_r2_deg_metric,
        title_prefix=f"R2 ({selected_deg_label})",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    for pds_metric in ["pds_cosine", "pds_l2", "pds_l1"]:
        plot_pds_vs_parameter(
            data=performance_data,
            save_path=(plot_dir / f"{pds_metric}_vs_control_bias.svg"),
            x_column="B",
            title=r"$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{β}$  $\mathbf{(Simulation)}$",
            x_label="Control Bias (β)",
            window=window,
            pds_metric=pds_metric,
            log_x=False,
            use_loess=use_loess,
        )
        plot_pds_vs_parameter(
            data=performance_data,
            save_path=(plot_dir / f"{pds_metric}_vs_n0.svg"),
            x_column="N0",
            title=r"$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{n_0}$ $\mathbf{(Simulation)}$",
            x_label="Number of Control Cells ($n_0$)",
            window=window,
            pds_metric=pds_metric,
            log_x=True,
            use_loess=use_loess,
        )
        plot_pds_vs_parameter(
            data=performance_data,
            save_path=(plot_dir / f"{pds_metric}_vs_perturbations.svg"),
            x_column="P",
            title=r"$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{k}$ $\mathbf{(Simulation)}$",
            x_label="Number of Perturbations ($P$)",
            window=window,
            pds_metric=pds_metric,
            log_x_if_range=True,
            use_loess=use_loess,
        )
        plot_pds_vs_parameter(
            data=performance_data,
            save_path=(plot_dir / f"{pds_metric}_vs_sparsity.svg"),
            x_column="sparsity",
            title=r"$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{Sparsity}$  $\mathbf{(Simulation)}$",
            x_label="Sparsity",
            window=window,
            pds_metric=pds_metric,
            log_x=False,
            use_loess=use_loess,
        )
        plot_pds_vs_parameter(
            data=performance_data,
            save_path=(plot_dir / f"{pds_metric}_vs_systematic_variation.svg"),
            x_column="systematic_variation",
            title=r"$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{Systematic}$ $\mathbf{Variation}$  $\mathbf{(Simulation)}$",
            x_label="Systematic Variation",
            window=window,
            pds_metric=pds_metric,
            log_x=False,
            use_loess=use_loess,
        )
        plot_pds_vs_parameter(
            data=performance_data,
            save_path=(plot_dir / f"{pds_metric}_vs_intra_corr.svg"),
            x_column="intra_corr",
            title=r"$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{Intra}$-$\mathbf{data}$ $\mathbf{Correlation}$  $\mathbf{(Simulation)}$",
            x_label="Intra-data Correlation",
            window=window,
            pds_metric=pds_metric,
            log_x=False,
            use_loess=use_loess,
        )

    histogram_specs = [
        ("sparsity", "sparsity.svg", "Sparsity", False),
        ("median_library_size", "median_library_size.svg", "Median Library Size", True),
        (
            "systematic_variation",
            "systematic_variation.svg",
            "Systematic Variation",
            False,
        ),
        ("intra_corr", "intra_corr.svg", "Intra-data Correlation", False),
        ("vendi_score", "vendi_score.svg", "Vendi Score", False),
    ]
    for column, filename, label, use_log_x in histogram_specs:
        if column not in statistics_data.columns:
            print(f"Skipping histogram for '{column}': column not found.")
            continue
        plot_statistics(
            data=statistics_data,
            save_path=(plot_dir / filename),
            column=column,
            x_label=label,
            title=f"Histogram of {label}",
            log_x=use_log_x,
        )

    plot_metric_vs_parameter(
        data=performance_data,
        save_path=(plot_dir / "mse_vs_p_effect.svg"),
        x_column="p_effect",
        y_column="mse",
        title=r"$\mathbf{MSE}$ $\mathbf{(Simulation)}$",
        x_label=r"Perturbation Probability ($\delta$)",
        window=window,
        color_by_model=True,
        use_loess=use_loess,
    )
    plot_metric_vs_parameter(
        data=statistics_data,
        save_path=(plot_dir / "systematic_variation_vs_intra_corr.svg"),
        x_column="systematic_variation",
        y_column="intra_corr",
        title=r"$\mathbf{Intra}$-$\mathbf{data}$ $\mathbf{Correlation}$ $\mathbf{by}$ $\mathbf{Systematic}$ $\mathbf{Variation}$  $\mathbf{(Simulation)}$",
        x_label="Systematic Variation",
        window=window,
        use_loess=use_loess,
    )
    plot_metric_vs_parameter(
        data=statistics_data,
        save_path=(plot_dir / "systematic_variation_vs_vendi_score.svg"),
        x_column="systematic_variation",
        y_column="vendi_score",
        title=r"$\mathbf{Vendi}$ $\mathbf{Score}$ $\mathbf{by}$ $\mathbf{Systematic}$ $\mathbf{Variation}$  $\mathbf{(Simulation)}$",
        x_label="Systematic Variation",
        window=window,
        use_loess=use_loess,
    )
    plot_metric_vs_parameter(
        data=statistics_data,
        save_path=(plot_dir / "vendi_score_vs_perturbations.pdf"),
        x_column="P",
        y_column="vendi_score",
        title=r"$\mathbf{Vendi}$ $\mathbf{Score}$ $\mathbf{by}$ $\mathbf{k}$  $\mathbf{(Simulation)}$",
        x_label="Number of Perturbations ($P$)",
        window=window,
        log_x_if_range=False,
        use_loess=use_loess,
    )

    print(f"Successfully generated all plots from {args.results}")


if __name__ == "__main__":
    main()
