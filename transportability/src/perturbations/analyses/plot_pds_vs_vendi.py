"""Plot PDS metrics against the vendi-score ratio from a folder of results CSVs."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .plot_utils import (
    COMMON_METRIC_LABELS,
    apply_paper_plot_style,
    coerce_numeric,
    compute_axis_limits,
    get_available_model_order,
    get_model_colors,
    metric_axis_label,
    with_vendi_ratio,
)

PDS_METRICS: tuple[str, ...] = ("pds_l1", "pds_l2", "pds_cosine")
VENDI_RATIO_COLUMN = "vendi_ratio"
VENDI_RATIO_LABEL = "VR"
CONTEXT_SPECIFIC_BASELINE_MODELS: tuple[str, ...] = (
    "Context-Average",
    "Context-linearPCA",
)
EXCLUDED_DATASETS: tuple[str, ...] = ("causalDGP", "naiveDGP")
EXCLUDED_MODELS: tuple[str, ...] = (
    "Control",
    "Average",
    "Context-Average",
    "Context-linearPCA",
)
DATASET_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "norman19": "Norman19",
    "replogle22": "Replogle22",
    "CD4+": "Zhu25",
    "directDGP": "DirectDGP",
    "causalDGP": "CausalDGP",
}
REQUIRED_COLUMNS: tuple[str, ...] = (
    *PDS_METRICS,
    "dataset",
    "model",
    "vendi_score_pred",
    "vendi_score_obs",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    *PDS_METRICS,
    "vendi_score_pred",
    "vendi_score_obs",
)
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results" / "pds_vs_vendi" / "in_context_only"
)
DEFAULT_OUTPUT_DIRNAME = "plots"
FIG_SIZE = (8.0, 5.5)

apply_paper_plot_style()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read all result CSV files in a folder and plot each PDS metric "
            "against the vendi-score ratio."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing results CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where plots will be written. Defaults to <results_dir>/plots.",
    )
    parser.add_argument(
        "--drop_context_specific_baselines",
        action="store_true",
        help="Drop Context-Average and Context-linearPCA from all plots.",
    )
    return parser.parse_args(argv)


def resolve_results_dir(results_dir: str) -> Path:
    """
    Resolve and validate the input results directory.

    Args:
        results_dir: Input directory path.

    Returns:
        Resolved directory path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    path = Path(results_dir)
    if not path.exists():
        raise FileNotFoundError(f"Results directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {path}")
    return path


def resolve_output_dir(results_dir: Path, output_dir: str | None) -> Path:
    """
    Resolve and create the output directory.

    Args:
        results_dir: Directory containing the source results.
        output_dir: Optional user-specified output directory.

    Returns:
        Output directory path.
    """
    path = results_dir / DEFAULT_OUTPUT_DIRNAME if output_dir is None else Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_results_paths(results_dir: Path) -> list[Path]:
    """
    Return all CSV result files in a directory.

    Args:
        results_dir: Directory containing CSV files.

    Returns:
        Sorted CSV paths.

    Raises:
        FileNotFoundError: If the directory contains no CSV files.
    """
    csv_paths = sorted(results_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    return csv_paths


def load_results_tables(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """
    Load and validate all result CSVs.

    Args:
        csv_paths: CSV files to load.

    Returns:
        Concatenated results table.

    Raises:
        ValueError: If any file is missing required columns.
    """
    data_frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        if "dataset" not in data.columns:
            # Derive dataset name from filename: strip "results_" prefix and timestamp/suffix.
            stem = csv_path.stem.removeprefix("results_")
            # Remove trailing _YYYYMMDD_HHMMSS timestamp if present.
            stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
            data["dataset"] = stem
        missing_columns = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
        if missing_columns:
            raise ValueError(
                f"Results file {csv_path} is missing required columns: {missing_columns}"
            )
        data_frames.append(data)
    return pd.concat(data_frames, ignore_index=True)


def prepare_results_data(
    data: pd.DataFrame,
    *,
    drop_context_specific_baselines: bool = False,
) -> pd.DataFrame:
    """
    Prepare raw results for plotting.

    Args:
        data: Concatenated results table.
        drop_context_specific_baselines: Whether to remove context-specific
            baseline models before plotting.

    Returns:
        Cleaned results with a vendi-ratio column added.

    Raises:
        ValueError: If no valid rows remain after filtering.
    """
    prepared = data.copy()
    prepared["dataset"] = prepared["dataset"].astype("string").str.strip()
    prepared["model"] = prepared["model"].astype("string").str.strip()
    prepared = prepared[
        prepared["dataset"].notna()
        & (prepared["dataset"].str.len() > 0)
        & prepared["model"].notna()
        & (prepared["model"].str.len() > 0)
    ].copy()

    if "status" in prepared.columns:
        before = len(prepared)
        prepared = prepared[prepared["status"].astype(str).str.lower() == "success"].copy()
        print(f"Kept successful runs: {len(prepared)}/{before}")

    # Drop excluded datasets (e.g. causalDGP).
    excluded_ds_mask = prepared["dataset"].str.contains(
        "|".join(EXCLUDED_DATASETS), case=False, na=False
    )
    if excluded_ds_mask.any():
        before = len(prepared)
        prepared = prepared[~excluded_ds_mask].copy()
        print(f"Dropped excluded datasets: {len(prepared)}/{before} rows kept")

    # Drop excluded models.
    if EXCLUDED_MODELS:
        before = len(prepared)
        prepared = prepared[~prepared["model"].isin(EXCLUDED_MODELS)].copy()
        print(f"Dropped excluded models: {len(prepared)}/{before} rows kept")

    if drop_context_specific_baselines:
        before = len(prepared)
        prepared = prepared[~prepared["model"].isin(CONTEXT_SPECIFIC_BASELINE_MODELS)].copy()
        print(f"Dropped context-specific baselines: {len(prepared)}/{before} rows kept")

    if prepared.empty:
        raise ValueError("No valid rows available to plot.")

    coerce_numeric(prepared, NUMERIC_COLUMNS)
    prepared = with_vendi_ratio(prepared, ratio_column=VENDI_RATIO_COLUMN)
    return prepared


def paired_metric_data(data: pd.DataFrame, pds_metric: str) -> pd.DataFrame:
    """
    Return finite paired values for one PDS metric and vendi ratio.

    Args:
        data: Prepared results table.
        pds_metric: PDS metric column name.

    Returns:
        Dataframe with finite `model`, PDS metric, and vendi ratio values.

    Raises:
        ValueError: If no finite paired values are available.
    """
    plot_data = data.loc[:, ["dataset", "model", pds_metric, VENDI_RATIO_COLUMN]].copy()
    finite_mask = np.isfinite(plot_data[pds_metric].to_numpy(dtype=float)) & np.isfinite(
        plot_data[VENDI_RATIO_COLUMN].to_numpy(dtype=float)
    )
    plot_data = plot_data.loc[finite_mask].copy()
    if plot_data.empty:
        raise ValueError(f"No valid paired values found for {pds_metric}.")
    return plot_data


def get_available_dataset_order(
    data: pd.DataFrame,
    *,
    dataset_column: str = "dataset",
) -> list[str]:
    """Return datasets present in a dataframe using first-seen order."""
    datasets = data[dataset_column].dropna().astype("string").str.strip().tolist()
    return list(dict.fromkeys(dataset for dataset in datasets if dataset))


def get_dataset_markers(dataset_order: Sequence[str]) -> dict[str, str]:
    """Return one marker for each dataset using a stable marker cycle."""
    return {
        dataset_name: DATASET_MARKERS[index % len(DATASET_MARKERS)]
        for index, dataset_name in enumerate(dataset_order)
    }


def dataset_display_name(dataset_name: str) -> str:
    """Return the legend label for one dataset."""
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)


def save_pds_vs_vendi_plot(
    data: pd.DataFrame,
    *,
    pds_metric: str,
    output_path: Path,
) -> None:
    """
    Save one PDS-vs-vendi-ratio scatter plot.

    Args:
        data: Prepared results table.
        pds_metric: PDS metric to plot on the x-axis.
        output_path: Plot output path.
    """
    plot_data = paired_metric_data(data, pds_metric)
    dataset_order = get_available_dataset_order(plot_data)
    dataset_markers = get_dataset_markers(dataset_order)
    model_order = get_available_model_order(plot_data)
    model_colors = get_model_colors(model_order)
    plotted_models: list[str] = []
    plotted_datasets: list[str] = []
    x_limits = compute_axis_limits(plot_data[pds_metric].to_numpy(dtype=float))
    y_limits = compute_axis_limits(plot_data[VENDI_RATIO_COLUMN].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for model_name in model_order:
        for dataset_name in dataset_order:
            model_data = plot_data.loc[
                (plot_data["model"] == model_name) & (plot_data["dataset"] == dataset_name)
            ]
            if model_data.empty:
                continue
            x_vals = model_data[pds_metric].to_numpy(dtype=float)
            y_vals = model_data[VENDI_RATIO_COLUMN].to_numpy(dtype=float)
            ax.scatter(
                x_vals,
                y_vals,
                color=model_colors[model_name],
                marker=dataset_markers[dataset_name],
                alpha=0.6,
                s=90,
                linewidths=0.5,
                edgecolors="face",
            )

            # Median with 95th-percentile error bars.
            x_med = np.median(x_vals)
            y_med = np.median(y_vals)
            x_lo, x_hi = np.percentile(x_vals, [2.5, 97.5])
            y_lo, y_hi = np.percentile(y_vals, [2.5, 97.5])
            ax.errorbar(
                x_med,
                y_med,
                xerr=[[x_med - x_lo], [x_hi - x_med]],
                yerr=[[y_med - y_lo], [y_hi - y_med]],
                fmt=dataset_markers[dataset_name],
                color=model_colors[model_name],
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.2,
                elinewidth=1.2,
                capsize=3,
                zorder=5,
            )

            if model_name not in plotted_models:
                plotted_models.append(model_name)
            if dataset_name not in plotted_datasets:
                plotted_datasets.append(dataset_name)

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel(metric_axis_label(pds_metric, COMMON_METRIC_LABELS[pds_metric]))
    ax.set_ylabel(metric_axis_label(VENDI_RATIO_COLUMN, VENDI_RATIO_LABEL))
    ax.grid(True, linestyle="--", alpha=0.4)
    sns.despine(ax=ax)
    legend_handles: list[Patch | Line2D] = []
    if plotted_models:
        legend_handles.extend(
            [
                Patch(
                    facecolor=model_colors[model_name],
                    edgecolor="none",
                    label=model_name,
                )
                for model_name in plotted_models
            ]
        )
    if plotted_datasets:
        legend_handles.extend(
            [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=dataset_markers[dataset_name],
                    markersize=7,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=1.6,
                    label=dataset_display_name(dataset_name),
                )
                for dataset_name in plotted_datasets
            ]
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Model / Dataset",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    fig.subplots_adjust(left=0.12, bottom=0.14, top=0.96, right=0.74)
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Generated {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the PDS-vs-vendi plotting workflow.

    Args:
        argv: Optional CLI argument list.
    """
    args = parse_args(argv)
    results_dir = resolve_results_dir(args.results_dir)
    output_dir = resolve_output_dir(results_dir, args.output_dir)
    csv_paths = discover_results_paths(results_dir)
    print(f"Loading {len(csv_paths)} CSV files from {results_dir}")

    results = load_results_tables(csv_paths)
    prepared = prepare_results_data(
        results,
        drop_context_specific_baselines=args.drop_context_specific_baselines,
    )

    for pds_metric in PDS_METRICS:
        save_pds_vs_vendi_plot(
            prepared,
            pds_metric=pds_metric,
            output_path=output_dir / f"{pds_metric}_vs_vendi_ratio.pdf",
        )


if __name__ == "__main__":
    main()
