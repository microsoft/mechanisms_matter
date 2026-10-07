"""Plot Replogle22 cross-context minus in-context metric deltas."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ...analyses.plot_utils import (
    COMMON_METRIC_LABELS,
    MODEL_COLORS,
    apply_paper_plot_style,
    coerce_numeric,
    resolve_expected_model_order,
    with_vendi_ratio,
)

apply_paper_plot_style()

CELL_LINE_ORDER: tuple[str, ...] = ("RPE1", "HepG2", "Jurkat")
SPLIT_ORDER: tuple[str, ...] = ("in_context", "cross_context")
DEFAULT_RESULT_PATHS: dict[str, dict[str, Path]] = {
    "in_context": {
        "RPE1": Path("results/real_experiments/results_replogle22_RPE1_in-context.csv"),
        "HepG2": Path("results/real_experiments/results_replogle22_HepG2_in-context.csv"),
        "Jurkat": Path("results/real_experiments/results_replogle22_Jurkat_in-context.csv"),
    },
    "cross_context": {
        "RPE1": Path("results/real_experiments/results_replogle22_RPE1_cross-context.csv"),
        "HepG2": Path("results/real_experiments/results_replogle22_HepG2_cross-context.csv"),
        "Jurkat": Path("results/real_experiments/results_replogle22_Jurkat_cross-context.csv"),
    },
}
OUTPUT_DIR_NAME = "replogle22_cross_minus_in_context_delta_plots"
VENDI_RATIO_COLUMN = "vendi_score_ratio"
BAR_WIDTH = 0.58
BAR_EDGE_COLOR = "#2b2b2b"
BAR_EDGE_WIDTH = 1.1
ZERO_LINE_COLOR = "#333333"
ZERO_LINE_WIDTH = 1.4
GRID_ALPHA = 0.25
FIG_SIZE = (5.2, 4.3)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot Replogle22 mean performance deltas, computed as average "
            "cross-context score minus average in-context score."
        )
    )
    for split_key, split_paths in DEFAULT_RESULT_PATHS.items():
        for cell_line, default_path in split_paths.items():
            arg_cell_line = cell_line.lower()
            parser.add_argument(
                f"--{split_key}_{arg_cell_line}",
                type=Path,
                default=default_path,
                help=f"Path to the {split_key.replace('_', '-')} {cell_line} results CSV.",
            )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where model subfolders and metric plots will be saved.",
    )
    return parser.parse_args(argv)


def resolve_results_path(results_path: Path) -> Path:
    """Resolve a results CSV path and validate that it exists."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    return results_path


def resolve_result_paths(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    """Return validated input paths keyed by split and cell line."""
    return {
        split_key: {
            cell_line: resolve_results_path(getattr(args, f"{split_key}_{cell_line.lower()}"))
            for cell_line in CELL_LINE_ORDER
        }
        for split_key in SPLIT_ORDER
    }


def default_output_dir(result_paths: dict[str, dict[str, Path]]) -> Path:
    """Return the default output directory for the provided input files."""
    all_paths = [path for split_paths in result_paths.values() for path in split_paths.values()]
    common_parent = Path(os.path.commonpath([str(path.parent.absolute()) for path in all_paths]))
    return common_parent / OUTPUT_DIR_NAME


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


def load_results(result_paths: dict[str, dict[str, Path]]) -> pd.DataFrame:
    """Load all input CSVs into one annotated dataframe."""
    frames: list[pd.DataFrame] = []
    for split_key in SPLIT_ORDER:
        for cell_line in CELL_LINE_ORDER:
            path = result_paths[split_key][cell_line]
            print(f"Using {split_key} {cell_line} results: {path}")
            frame = prepare_results_data(pd.read_csv(path))
            frame["split"] = split_key
            frame["cell_line"] = cell_line
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["split"] = pd.Categorical(
        combined["split"],
        categories=SPLIT_ORDER,
        ordered=True,
    )
    combined["cell_line"] = pd.Categorical(
        combined["cell_line"],
        categories=CELL_LINE_ORDER,
        ordered=True,
    )
    return combined


def compute_mean_deltas(data: pd.DataFrame, model_order: Sequence[str]) -> pd.DataFrame:
    """Compute average cross-context minus average in-context scores."""
    means = (
        data.groupby(["split", "cell_line", "model"], observed=False)[list(METRICS)]
        .mean()
        .sort_index()
    )

    rows: list[dict[str, object]] = []
    for model in model_order:
        for metric in METRICS:
            for cell_line in CELL_LINE_ORDER:
                in_value = means.loc[("in_context", cell_line, model), metric]
                cross_value = means.loc[("cross_context", cell_line, model), metric]
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "cell_line": cell_line,
                        "delta": cross_value - in_value,
                    }
                )
    delta_data = pd.DataFrame(rows)
    delta_data["cell_line"] = pd.Categorical(
        delta_data["cell_line"],
        categories=CELL_LINE_ORDER,
        ordered=True,
    )
    return delta_data


def safe_path_component(value: str) -> str:
    """Return a filesystem-safe path component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def y_limits(values: np.ndarray) -> tuple[float, float]:
    """Return y-limits padded around finite values and zero."""
    finite_values = values[np.isfinite(values)]
    y_min = float(min(np.min(finite_values), 0.0))
    y_max = float(max(np.max(finite_values), 0.0))
    if np.isclose(y_min, y_max):
        padding = 0.05 * max(abs(y_min), 1.0)
        return y_min - padding, y_max + padding

    padding = 0.08 * (y_max - y_min)
    return y_min - padding, y_max + padding


def save_delta_plot(
    metric_delta_data: pd.DataFrame,
    *,
    model: str,
    metric: str,
    output_path: Path,
) -> bool:
    """Save one single-panel bar plot for a model and metric."""
    plot_data = metric_delta_data.sort_values("cell_line")
    delta_values = plot_data["delta"].to_numpy(dtype=float)
    finite = np.isfinite(delta_values)
    if not finite.any():
        return False

    x_positions = np.arange(len(CELL_LINE_ORDER), dtype=float)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(
        x_positions[finite],
        delta_values[finite],
        width=BAR_WIDTH,
        color=MODEL_COLORS[model],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(CELL_LINE_ORDER)
    ax.set_xlim(-0.5, len(CELL_LINE_ORDER) - 0.5)
    ax.set_ylim(*y_limits(delta_values))
    ax.set_xlabel("Cell line")
    ax.set_ylabel(f"delta ({COMMON_METRIC_LABELS[metric]})")
    ax.grid(axis="y", alpha=GRID_ALPHA, zorder=1)
    fig.subplots_adjust(left=0.22, bottom=0.18, top=0.96, right=0.96)
    fig.savefig(output_path, bbox_inches=None)
    plt.close(fig)
    return True


def save_all_delta_plots(
    delta_data: pd.DataFrame,
    model_order: Sequence[str],
    output_dir: Path,
) -> tuple[int, list[tuple[str, str]]]:
    """Save all model/metric delta plots into model-specific subfolders."""
    generated_count = 0
    skipped: list[tuple[str, str]] = []

    for model in model_order:
        model_output_dir = output_dir / safe_path_component(model)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_data = delta_data.loc[delta_data["model"] == model]

        for metric in METRICS:
            metric_delta_data = model_data.loc[model_data["metric"] == metric]
            output_path = model_output_dir / f"{metric}.svg"
            if save_delta_plot(
                metric_delta_data,
                model=model,
                metric=metric,
                output_path=output_path,
            ):
                print(f"Saved: {output_path}")
                generated_count += 1
            else:
                print(f"Skipping {model} {metric}: no valid delta values to plot.")
                skipped.append((model, metric))

    return generated_count, skipped


def main() -> None:
    """Generate Replogle22 model-metric delta bar plots."""
    args = parse_args()
    result_paths = resolve_result_paths(args)
    results_data = load_results(result_paths)
    model_order = resolve_expected_model_order(results_data)
    print(f"Models available across inputs: {model_order}")

    output_dir = (
        args.output_dir if args.output_dir is not None else default_output_dir(result_paths)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    delta_data = compute_mean_deltas(results_data, model_order)
    generated_count, skipped = save_all_delta_plots(delta_data, model_order, output_dir)

    print(f"Generated {generated_count} plot(s) in {output_dir}")
    if skipped:
        skipped_labels = [f"{model}:{metric}" for model, metric in skipped]
        print(f"Skipped model/metric pairs: {', '.join(skipped_labels)}")


if __name__ == "__main__":
    main()
