"""Plot styling and helper utilities for perturbation analysis figures."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from scipy import stats

PAPER_PLOT_RC_PARAMS: dict[str, Any] = {
    "figure.facecolor": "white",
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif"],
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 21,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 16,
    "legend.title_fontsize": 15,
    "lines.linewidth": 3.0,
    "patch.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}
EXPECTED_MODEL_ORDER: tuple[str, ...] = (
    "Control",
    "Average",
    "Context-Average",
    "linearPCA",
    "Context-linearPCA",
    "scVI",
    "GEARS",
    "CPA",
    "STATE",
    "STATE-Geneformer",
    "scLDM",
)
MODEL_TICK_LABEL_ROTATION = 45.0
MODEL_TICK_LABEL_ALIGNMENT = "right"
MODEL_COLORS: dict[str, str] = {
    "Control": "#4e79a7",
    "Average": "#f28e2b",
    "Context-Average": "#e15759",
    "linearPCA": "#59a14f",
    "Context-linearPCA": "#7bb661",
    "scVI": "#76b7b2",
    "GEARS": "#9c755f",
    "CPA": "#3c5488",
    "STATE": "#b07aa1",
    "STATE-Geneformer": "#7f7f7f",
    "scLDM": "#d4a017",
}
COMMON_METRIC_LABELS: dict[str, str] = {
    "pearson": "Pearson",
    "pearson_degs": "Pearson (DEGs)",
    "pearson_true_degs": "Pearson (Affected Genes)",
    "mae": "MAE",
    "mae_degs": "MAE (DEGs)",
    "mae_true_degs": "MAE (Affected Genes)",
    "mse": "MSE",
    "mse_degs": "MSE (DEGs)",
    "mse_true_degs": "MSE (Affected Genes)",
    "r2": "R2",
    "r2_degs": "R2 (DEGs)",
    "r2_true_degs": "R2 (Affected Genes)",
    "parametric_distance": "Parametric Distance",
    "mmd_distance": "MMD",
    "fid_distance": "FD Distance",
    "des_recall": "DES (Recall)",
    "des_precision": "DES (Precision)",
    "des_jaccard": "DES (Jaccard)",
    "pds_l1": "PDS (L1)",
    "pds_l2": "PDS (L2)",
    "pds_cosine": "PDS (Cosine)",
    "vendi_ratio": "Vendi Ratio",
    "vendi_score_ratio": "Vendi Ratio",
}
COMMON_METRIC_BASE_COLORS: dict[str, str] = {
    "pearson": "#c92a2a",
    "pearson_degs": "#1c7ed6",
    "pearson_true_degs": "#1c7ed6",
    "mae": "#c92a2a",
    "mae_degs": "#1c7ed6",
    "mae_true_degs": "#1c7ed6",
    "mse": "#c92a2a",
    "mse_degs": "#1c7ed6",
    "mse_true_degs": "#1c7ed6",
    "r2": "#c92a2a",
    "r2_degs": "#1c7ed6",
    "r2_true_degs": "#1c7ed6",
    "parametric_distance": "#e67700",
    "mmd_distance": "#0b7285",
    "fid_distance": "#c2255c",
    "des_recall": "#5f3dc4",
    "des_precision": "#2f9e44",
    "des_jaccard": "#1971c2",
    "pds_l1": "#c92a2a",
    "pds_l2": "#1c7ed6",
    "pds_cosine": "#2b8a3e",
    "vendi_ratio": "#9c36b5",
    "vendi_score_ratio": "#9c36b5",
}

R2_METRIC_PREFIX = "r2"
R2_AXIS_LABEL = r"$-\log(1 - R^2)$"
R2_PLOT_UPPER_BOUND = np.nextafter(1.0, 0.0)
PlotValueTransform = Callable[[str, np.ndarray], np.ndarray]


def apply_paper_plot_style(extra_rc_params: Mapping[str, Any] | None = None) -> None:
    """Apply a shared paper-style Matplotlib/Seaborn theme."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper")
    sns.set_style("ticks")

    rc_params = dict(PAPER_PLOT_RC_PARAMS)
    if extra_rc_params:
        rc_params.update(extra_rc_params)
    plt.rcParams.update(rc_params)


def coerce_numeric(data: pd.DataFrame, columns: Iterable[str]) -> None:
    """Convert matching columns to numeric in place."""
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")


def get_available_model_order(
    data: pd.DataFrame,
    *,
    model_column: str = "model",
    expected_order: Sequence[str] = EXPECTED_MODEL_ORDER,
) -> list[str]:
    """Return models present in a dataframe using a canonical ordering."""
    present = set(data[model_column].dropna().astype(str).tolist())
    ordered = [model for model in expected_order if model in present]
    extras = sorted(present.difference(expected_order))
    return [*ordered, *extras]


def resolve_expected_model_order(
    *data_frames: pd.DataFrame,
    model_column: str = "model",
    expected_order: Sequence[str] = EXPECTED_MODEL_ORDER,
    require_all_expected: bool = False,
) -> list[str]:
    """Return a strict canonical model order for one or more result tables."""
    discovered_models = {
        model for data in data_frames for model in data[model_column].dropna().astype(str).tolist()
    }
    unexpected_models = sorted(discovered_models.difference(expected_order))
    if unexpected_models:
        raise ValueError(f"Unexpected model names found in results: {unexpected_models}")

    ordered_models = [model for model in expected_order if model in discovered_models]
    if require_all_expected:
        missing_models = [model for model in expected_order if model not in discovered_models]
        if missing_models:
            raise ValueError(f"Expected model names missing from results: {missing_models}")
        return list(expected_order)

    if not ordered_models:
        raise ValueError("No known models found in the provided results files.")
    return ordered_models


def get_model_colors(
    model_order: Sequence[str],
    *,
    base_colors: Mapping[str, Any] = MODEL_COLORS,
) -> dict[str, Any]:
    """Return one color for each model, with palette fallbacks for extras."""
    fallback_palette = sns.color_palette("tab10", n_colors=max(len(model_order), 1))
    return {
        model_name: base_colors.get(
            model_name,
            fallback_palette[index % len(fallback_palette)],
        )
        for index, model_name in enumerate(model_order)
    }


def moving_average(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sliding-window average of ``y`` over sorted ``x``."""
    sorted_index = np.argsort(x)
    x_sorted = x[sorted_index]
    y_sorted = y[sorted_index]

    x_average = []
    y_average = []
    for index in range(len(x_sorted) - window + 1):
        x_average.append(np.mean(x_sorted[index : index + window]))
        y_average.append(np.mean(y_sorted[index : index + window]))

    return np.asarray(x_average), np.asarray(y_average)


def effective_moving_average_window(
    n_points: int,
    requested_window: int,
) -> int | None:
    """Pick a valid moving-average window for the available points."""
    if n_points < 2:
        return None
    if n_points >= requested_window:
        return int(requested_window)
    return max(2, n_points // 3)


def loess_trend_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a robust LOESS-smoothed trend line using a window-matched span."""
    from skmisc.loess import loess

    n_points = len(x_values)
    effective_window = effective_moving_average_window(n_points, window)
    if effective_window is None or n_points < 3:
        return None

    smoothed_input = (
        pd.DataFrame({"x": x_values, "y": y_values})
        .groupby("x", as_index=False, sort=True)["y"]
        .mean()
    )
    x_unique = smoothed_input["x"].to_numpy(dtype=float)
    y_unique = smoothed_input["y"].to_numpy(dtype=float)
    if len(x_unique) < 3:
        return None

    base_span = min(1.0, max(effective_window / len(x_unique), 3 / len(x_unique)))
    candidate_spans = []
    for span in (
        base_span,
        max(base_span, 0.30),
        max(base_span, 0.40),
        max(base_span, 0.50),
        max(base_span, 0.65),
        max(base_span, 0.80),
        1.0,
    ):
        if span not in candidate_spans:
            candidate_spans.append(span)

    predictions = None
    for span in candidate_spans:
        try:
            fitted_model = loess(
                x_unique,
                y_unique,
                span=span,
                degree=1,
                family="symmetric",
            )
            fitted_model.fit()
            predictions = fitted_model.predict(x_unique, stderror=False).values
            break
        except ValueError:
            continue

    if predictions is None:
        return None

    finite_mask = np.isfinite(predictions)
    if not finite_mask.any():
        return None
    return x_unique[finite_mask], predictions[finite_mask]


def compute_trend_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    window: int,
    use_loess: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return either a LOESS or moving-average trend line."""
    if use_loess:
        return loess_trend_line(
            np.asarray(x_values, dtype=float),
            np.asarray(y_values, dtype=float),
            window=window,
        )

    effective_window = effective_moving_average_window(len(x_values), window)
    if effective_window is None:
        return None
    return moving_average(
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        window=effective_window,
    )


def has_valid_metric(data: pd.DataFrame, metric: str) -> bool:
    """Return whether a metric column contains at least one finite value."""
    if metric not in data.columns:
        return False
    values = transform_metric_values(
        pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float),
        metric,
    )
    return bool(np.isfinite(values).any())


def is_r2_metric(metric: str) -> bool:
    """Return whether a metric should use the shared R2 plot transform."""
    return metric == R2_METRIC_PREFIX or metric.startswith(f"{R2_METRIC_PREFIX}_")


def transform_r2_values(values: np.ndarray) -> np.ndarray:
    """Map R2 values to the shared plot scale -log(1 - R2)."""
    numeric_values = np.asarray(values, dtype=float)
    transformed = np.full(numeric_values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(numeric_values)
    if not finite_mask.any():
        return transformed

    clipped_values = np.minimum(numeric_values[finite_mask], R2_PLOT_UPPER_BOUND)
    transformed[finite_mask] = -np.log1p(-clipped_values)
    return transformed


def transform_metric_values(values: np.ndarray, metric: str) -> np.ndarray:
    """Return plot-space values for a metric."""
    numeric_values = np.asarray(values, dtype=float)
    if is_r2_metric(metric):
        return transform_r2_values(numeric_values)
    return numeric_values


def transform_metric_bounds(bounds: tuple[float, float], metric: str) -> tuple[float, float]:
    """Return plot-space axis bounds for a metric."""
    transformed_bounds = transform_metric_values(np.asarray(bounds, dtype=float), metric)
    lower, upper = transformed_bounds.tolist()
    return float(min(lower, upper)), float(max(lower, upper))


def metric_axis_label(metric: str, default_label: str) -> str:
    """Return the axis label for one metric on the current plot scale."""
    if not is_r2_metric(metric):
        return default_label
    if default_label in {"", "Score"}:
        return R2_AXIS_LABEL
    if "$R^2$" in default_label:
        return default_label.replace("$R^2$", R2_AXIS_LABEL, 1)
    if "R2" in default_label:
        return default_label.replace("R2", R2_AXIS_LABEL, 1)
    return f"{default_label} ({R2_AXIS_LABEL} scale)"


def metrics_axis_label(metrics: Sequence[str], default_label: str) -> str:
    """Return a shared axis label for one or more plotted metrics."""
    if metrics and all(is_r2_metric(metric) for metric in metrics):
        return metric_axis_label(metrics[0], default_label)
    return default_label


def compute_pearson_correlation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    log_x: bool = False,
) -> tuple[float, float]:
    """Return Pearson correlation statistics for paired numeric arrays."""
    x_numeric = np.asarray(x_values, dtype=float)
    y_numeric = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x_numeric) & np.isfinite(y_numeric)
    x_numeric = x_numeric[finite_mask]
    y_numeric = y_numeric[finite_mask]

    if log_x:
        positive_mask = x_numeric > 0
        x_numeric = x_numeric[positive_mask]
        y_numeric = y_numeric[positive_mask]
        if x_numeric.size:
            x_numeric = np.log10(x_numeric)

    if x_numeric.size < 2:
        return np.nan, np.nan
    return stats.pearsonr(x_numeric, y_numeric)


def add_pearson_correlation_annotation(
    axis: Axes,
    corr: float,
    p_value: float,
    *,
    x: float = 0.05,
    y: float = 0.97,
    fontsize: float = 18,
) -> None:
    """Add a standard Pearson correlation annotation to one axis."""
    axis.text(
        x,
        y,
        f"Pearson R={corr:.2f}, P={p_value:.2e}",
        transform=axis.transAxes,
        fontsize=fontsize,
        va="bottom",
        ha="left",
    )


def default_plot_value_transform(metric: str, values: np.ndarray) -> np.ndarray:
    """Return plot-space values using the shared metric transform rules."""
    return transform_metric_values(values, metric)


def with_vendi_ratio(
    data: pd.DataFrame,
    *,
    pred_column: str = "vendi_score_pred",
    obs_column: str = "vendi_score_obs",
    ratio_column: str = "vendi_ratio",
) -> pd.DataFrame:
    """Add a vendi ratio column when the required source columns are present."""
    required = (pred_column, obs_column)
    if any(column not in data.columns for column in required):
        return data

    vendi_data = data.copy()
    obs = vendi_data[obs_column].to_numpy(dtype=float)
    vendi_data[ratio_column] = np.divide(
        vendi_data[pred_column].to_numpy(dtype=float),
        obs,
        out=np.full(len(vendi_data), np.nan, dtype=float),
        where=np.isfinite(obs) & (obs != 0),
    )
    return vendi_data


def expand_ylim(y_min: float, y_max: float, pad_fraction: float = 0.05) -> tuple[float, float]:
    """Pad y-axis limits while handling constant-valued series."""
    if np.isclose(y_min, y_max):
        pad = pad_fraction * max(abs(y_min), 1.0)
        return y_min - pad, y_max + pad

    pad = (y_max - y_min) * pad_fraction
    return y_min - pad, y_max + pad


def compute_axis_limits(
    values: np.ndarray,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Return padded finite axis limits, optionally clipped to bounds."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None

    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if bounds is not None:
        lower = max(bounds[0], lower)
        upper = min(bounds[1], upper)

    lower, upper = expand_ylim(lower, upper)
    if bounds is not None:
        lower = max(bounds[0], lower)
        upper = min(bounds[1], upper)
    return lower, upper


def compute_log_axis_limits(
    values: np.ndarray,
    *,
    pad_fraction: float = 0.05,
) -> tuple[float, float] | None:
    """Return padded positive axis limits for a log-scaled axis."""
    positive_values = np.asarray(values, dtype=float)
    positive_values = positive_values[np.isfinite(positive_values) & (positive_values > 0)]
    if positive_values.size == 0:
        return None

    lower = float(np.min(positive_values))
    upper = float(np.max(positive_values))
    if np.isclose(lower, upper):
        return lower / 1.25, upper * 1.25

    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    pad = (log_upper - log_lower) * pad_fraction
    return 10 ** (log_lower - pad), 10 ** (log_upper + pad)


def compute_metric_limits(
    data_frames: Sequence[pd.DataFrame],
    metrics: Iterable[str],
) -> dict[str, tuple[float, float]]:
    """Compute per-metric min/max bounds across one or more result tables."""
    limits: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        merged_values: list[np.ndarray] = []
        for data in data_frames:
            if metric not in data.columns:
                continue
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
            transformed_values = transform_metric_values(values, metric)
            finite_values = transformed_values[np.isfinite(transformed_values)]
            if finite_values.size:
                merged_values.append(finite_values)

        if not merged_values:
            continue

        merged = np.concatenate(merged_values)
        limits[metric] = (float(np.min(merged)), float(np.max(merged)))
    return limits


def get_plot_ylim(
    metrics: Sequence[str],
    metric_limits: Mapping[str, tuple[float, float]] | None,
) -> tuple[float, float] | None:
    """Resolve a shared y-axis range for one or more metrics."""
    if not metric_limits:
        return None

    bounds = [metric_limits[metric] for metric in metrics if metric in metric_limits]
    if not bounds:
        return None

    return expand_ylim(
        min(lower for lower, _ in bounds),
        max(upper for _, upper in bounds),
    )


def lighten(color: tuple[float, float, float], amount: float = 0.6) -> tuple[float, float, float]:
    """Blend a color toward white."""
    return tuple((1.0 - amount) * component + amount for component in color)


def darken(color: tuple[float, float, float], amount: float = 0.7) -> tuple[float, float, float]:
    """Blend a color toward black."""
    return tuple(component * amount for component in color)


def build_metric_styles(
    metrics: Sequence[str],
    metric_base_colors: Mapping[str, str],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Build consistent box/edge/dot colors for each metric."""
    cmap = plt.get_cmap("tab10")
    styles: dict[str, dict[str, tuple[float, float, float]]] = {}
    for index, metric in enumerate(metrics):
        base = metric_base_colors.get(metric, cmap(index % 10))
        base_rgb = mcolors.to_rgb(base)
        styles[metric] = {
            "box": lighten(base_rgb, amount=0.62),
            "edge": darken(base_rgb, amount=0.68),
            "dot": darken(base_rgb, amount=0.55),
        }
    return styles


def metric_legend_handles(
    metrics: Sequence[str],
    metric_styles: Mapping[str, Mapping[str, tuple[float, float, float]]],
    metric_labels: Mapping[str, str],
) -> list[Patch]:
    """Build legend patches for the provided metrics."""
    return [
        Patch(
            facecolor=metric_styles[metric]["box"],
            edgecolor=metric_styles[metric]["edge"],
            label=metric_labels.get(metric, metric),
        )
        for metric in metrics
    ]


def plot_model_grouped_boxplot(
    ax: Axes,
    data: pd.DataFrame,
    metrics: Sequence[str],
    model_order: Sequence[str],
    metric_styles: Mapping[str, Mapping[str, tuple[float, float, float]]],
    cluster_gap: float,
    ylabel: str,
    y_limits: tuple[float, float] | None = None,
    no_model_message: str | None = "No models available",
    x_tick_label_rotation: float = 0.0,
    x_tick_label_ha: str = "center",
    value_transform: PlotValueTransform = default_plot_value_transform,
) -> None:
    """Draw grouped boxplots with consistent styling and jittered points."""
    if not metrics:
        ax.text(
            0.5, 0.5, "No valid metrics to plot", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        return

    n_models = len(model_order)
    n_metrics = len(metrics)
    if n_models == 0:
        if no_model_message is None:
            raise ValueError("No models available for plotting.")
        ax.text(0.5, 0.5, no_model_message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    rng = np.random.default_rng(0)
    model_centers: list[float] = []
    any_boxes = False

    for model_index, model in enumerate(model_order):
        cluster_start = model_index * (n_metrics + cluster_gap)
        cluster_positions: list[float] = []

        for metric_index, metric in enumerate(metrics):
            position = cluster_start + metric_index
            raw_values = pd.to_numeric(
                data.loc[data["model"] == model, metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = value_transform(metric, raw_values)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            any_boxes = True
            style = metric_styles[metric]
            boxplot = ax.boxplot(
                values,
                positions=[position],
                widths=0.72,
                patch_artist=True,
                showfliers=False,
                zorder=2,
            )
            for box in boxplot["boxes"]:
                box.set_facecolor(style["box"])
                box.set_edgecolor(style["edge"])
                box.set_linewidth(2.0)
                box.set_alpha(0.9)
            for whisker in boxplot["whiskers"]:
                whisker.set_color(style["edge"])
                whisker.set_linewidth(1.9)
            for cap in boxplot["caps"]:
                cap.set_color(style["edge"])
                cap.set_linewidth(1.9)
            for median in boxplot["medians"]:
                median.set_color(style["dot"])
                median.set_linewidth(2.6)

            jitter = rng.normal(0.0, 0.05, size=values.size)
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                color=style["dot"],
                alpha=0.65,
                s=18,
                linewidths=0,
                zorder=3,
            )
            cluster_positions.append(position)

        if cluster_positions:
            model_centers.append(float(np.mean(cluster_positions)))
        else:
            model_centers.append(cluster_start + (n_metrics - 1) / 2.0)

    if not any_boxes:
        ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    total_width = (n_models - 1) * (n_metrics + cluster_gap) + n_metrics
    ax.set_xlim(-0.9, total_width - 0.1)
    ax.set_xticks(model_centers)
    ax.set_xticklabels(model_order)
    tick_label_style: dict[str, Any] = {
        "rotation": x_tick_label_rotation,
        "ha": x_tick_label_ha,
    }
    if x_tick_label_rotation:
        tick_label_style["rotation_mode"] = "anchor"
    plt.setp(ax.get_xticklabels(), **tick_label_style)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(axis="y", alpha=0.25)


def save_metric_group_boxplot(
    *,
    data: pd.DataFrame,
    model_order: Sequence[str],
    metrics: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
    metric_labels: Mapping[str, str],
    metric_base_colors: Mapping[str, str],
    cluster_gap: float = 1.2,
    fig_size: tuple[float, float] = (8.0, 5.8),
    metric_limits: Mapping[str, tuple[float, float]] | None = None,
    dpi: int | None = None,
    no_model_message: str | None = "No models available",
    x_tick_label_rotation: float = 0.0,
    x_tick_label_ha: str = "center",
    value_transform: PlotValueTransform = default_plot_value_transform,
) -> bool:
    """Render and save a grouped metric boxplot."""
    valid_metrics = [metric for metric in metrics if has_valid_metric(data, metric)]
    if not valid_metrics:
        return False

    metric_styles = build_metric_styles(valid_metrics, metric_base_colors)
    fig, ax = plt.subplots(figsize=fig_size)
    plot_model_grouped_boxplot(
        ax=ax,
        data=data,
        metrics=valid_metrics,
        model_order=model_order,
        metric_styles=metric_styles,
        cluster_gap=cluster_gap,
        ylabel=metrics_axis_label(valid_metrics, ylabel),
        y_limits=get_plot_ylim(valid_metrics, metric_limits),
        no_model_message=no_model_message,
        x_tick_label_rotation=x_tick_label_rotation,
        x_tick_label_ha=x_tick_label_ha,
        value_transform=value_transform,
    )
    ax.legend(
        handles=metric_legend_handles(valid_metrics, metric_styles, metric_labels),
        title="Metric",
        loc="best",
        frameon=True,
    )
    fig.tight_layout()

    save_kwargs: dict[str, Any] = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(output_path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    return True
