"""Plot a histogram of observed DEG ratios from observed-DEG count results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..plot_utils import apply_paper_plot_style

apply_paper_plot_style()


def compute_deg_ratios(results_path: Path) -> pd.Series:
    """Load a results CSV and return the validated observed DEG ratios."""
    data = pd.read_csv(results_path)

    observed_deg_counts = pd.to_numeric(data["n_obs_degs"], errors="coerce")
    n_genes = pd.to_numeric(data["n_genes"], errors="coerce")

    ratios = pd.Series(
        np.divide(
            observed_deg_counts.to_numpy(dtype=float),
            n_genes.to_numpy(dtype=float),
        ),
        index=data.index,
        name="deg_ratio",
    )
    return ratios


def plot_deg_ratio_histogram(
    deg_ratios: pd.Series,
    output_path: Path,
    bins: int = 30,
) -> None:
    """Plot and save a histogram of observed DEG ratios."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(deg_ratios.to_numpy(dtype=float), bins=bins, color="skyblue", edgecolor="black")
    ax.set_xlabel("Observed DEG ratio")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of observed DEG ratios")
    ax.set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot a histogram of observed DEG ratios from observed DEG count results."
    )
    parser.add_argument(
        "--results",
        type=str,
        help=(
            "Path to an observed DEG count CSV. If omitted, the latest file in "
            "results/observed_deg_counts is used."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30).",
    )
    return parser


def main() -> None:
    """Run the DEG-ratio histogram CLI."""
    args = build_arg_parser().parse_args()
    if args.results:
        results_path = Path(args.results)
    else:
        raise ValueError("Please provide a results CSV path with --results.")
    output_path = results_path.with_name(f"{results_path.stem}_deg_ratio_hist.png")
    deg_ratios = compute_deg_ratios(results_path)

    print(f"Using results: {results_path}")
    plot_deg_ratio_histogram(
        deg_ratios=deg_ratios,
        output_path=output_path,
        bins=args.bins,
    )
    print(f"Saved histogram to: {output_path}")


if __name__ == "__main__":
    main()
