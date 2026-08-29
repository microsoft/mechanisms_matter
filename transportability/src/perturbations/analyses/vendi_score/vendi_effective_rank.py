r"""
Validate Vendi score effective-rank tracking with DirectDGP.

Generates 100 randomly simulated datasets using DirectDGP-specific parameters
sampled from ``PARAM_RANGES`` (including P uniformly from 20 to 50) and plots
scatter plots of Vendi (cell + pseudobulk) and PDS-L1 vs P, reporting
Pearson r and p-value with OLS trend lines. It also compares cell and
pseudobulk Vendi using Spearman rank correlation.

Example:
    python -m perturbations.analyses.vendi_score.vendi_effective_rank \
        --output-dir results/vendi_effective_rank \
        --n-trials 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from perturbations.analyses.common import NORM_LAYER_KEY
from perturbations.analyses.plot_utils import apply_paper_plot_style
from perturbations.analyses.vendi_score.run_vendi import (
    _estimate_vendi_params,
    _pseudobulk_vendi,
    _split_half_pds,
)
from perturbations.data.dgp.directDGP import directDGP
from perturbations.metrics.reconstruction.vendi_score import vendi_score

from ..synthetic_simulations.sampling import (
    PARAM_RANGES,
    load_parameter_estimation_inputs,
    sample_parameters,
)

matplotlib.use("Agg")

_DEFAULT_OUTPUT_DIR = "results/vendi_effective_rank"
_CONTROL_LABEL = "control"
_DEFAULT_N_TRIALS = 100
_MIN_MARKER_AREA = 30.0
_MAX_MARKER_AREA = 140.0


def _scatter_encodings(
    results: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, LogNorm, list[Line2D]]:
    """Return shared color and marker-size encodings for all panels."""
    p_effect = results["p_effect"].to_numpy(dtype=float)
    positive_p_effect = p_effect[np.isfinite(p_effect) & (p_effect > 0)]
    if positive_p_effect.size == 0:
        raise ValueError("p_effect must contain at least one positive finite value.")
    p_effect_min = float(positive_p_effect.min())
    p_effect_max = float(positive_p_effect.max())
    if np.isclose(p_effect_min, p_effect_max):
        p_effect_min /= 1.01
        p_effect_max *= 1.01
    color_norm = LogNorm(vmin=p_effect_min, vmax=p_effect_max)

    effect_factor = results["effect_factor"].to_numpy(dtype=float)
    finite_effect_factor = effect_factor[np.isfinite(effect_factor)]
    if finite_effect_factor.size == 0:
        raise ValueError("effect_factor must contain at least one finite value.")
    effect_min = float(finite_effect_factor.min())
    effect_max = float(finite_effect_factor.max())
    if np.isclose(effect_min, effect_max):
        marker_areas = np.full(effect_factor.shape, (_MIN_MARKER_AREA + _MAX_MARKER_AREA) / 2)
        legend_values = np.asarray([effect_min])
        legend_areas = np.asarray([marker_areas[0]])
    else:
        marker_areas = np.interp(
            effect_factor,
            (effect_min, effect_max),
            (_MIN_MARKER_AREA, _MAX_MARKER_AREA),
        )
        legend_values = np.linspace(effect_min, effect_max, 3)
        legend_areas = np.interp(
            legend_values,
            (effect_min, effect_max),
            (_MIN_MARKER_AREA, _MAX_MARKER_AREA),
        )

    size_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=np.sqrt(area),
            markerfacecolor="0.4",
            markeredgecolor="none",
            alpha=0.65,
            label=f"{value:.2g}",
        )
        for value, area in zip(legend_values, legend_areas, strict=True)
    ]
    return p_effect, marker_areas, color_norm, size_handles


def _add_plot_guides(
    ax: Axes,
    size_handles: list[Line2D],
    *,
    position: Literal["upper-left", "lower-left", "below-identity"] = "upper-left",
) -> None:
    """Stack marker-size and trend guides in a clear plot region."""
    line_handles, line_labels = ax.get_legend_handles_labels()
    if position == "upper-left":
        size_location = "upper left"
        line_location = "upper left"
        line_anchor = (0.0, 0.68)
    elif position == "lower-left":
        size_location = "lower left"
        line_location = "lower left"
        line_anchor = (0.0, 0.38)
    else:
        size_location = "lower right"
        line_location = "lower right"
        line_anchor = (1.0, 0.38)
    size_legend = ax.legend(
        handles=size_handles,
        title="Effect factor",
        loc=size_location,
    )
    ax.add_artist(size_legend)
    if line_handles:
        line_entries = sorted(
            zip(line_handles, line_labels, strict=True),
            key=lambda entry: not entry[1].startswith("OLS"),
        )
        ax.legend(
            [handle for handle, _ in line_entries],
            [label for _, label in line_entries],
            loc=line_location,
            bbox_to_anchor=line_anchor,
        )
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)


def _generate_and_score(
    P: int,
    G: int,
    N0: int,
    Nk: int,
    p_effect: float,
    effect_factor: float,
    B: float,
    mu_l: float,
    inputs: dict[str, np.ndarray],
    n_pca_components: int,
    seed: int,
) -> dict[str, object]:
    """Generate one DirectDGP dataset at P perturbations and score it."""
    adata, _ = directDGP(
        G=G,
        N0=N0,
        Nk=Nk,
        P=P,
        p_effect=p_effect,
        effect_factor=effect_factor,
        B=B,
        mu_l=mu_l,
        all_theta=inputs["all_theta"],
        control_mu=inputs["control_mu"],
        pert_mu=inputs["pert_mu"],
        gene_names=inputs["gene_names"],
        seed=seed,
        normalize=True,
        normalized_layer_key=NORM_LAYER_KEY,
    )

    layer_key = NORM_LAYER_KEY
    pca_model, gamma, outer_sigma_squared = _estimate_vendi_params(
        adata, layer_key, _CONTROL_LABEL, n_pca_components, seed
    )
    vs_cell = float(
        vendi_score(
            ac=adata,
            n_pca_components=n_pca_components,
            layer_key=layer_key,
            control_label=_CONTROL_LABEL,
            random_state=seed,
            gamma=gamma,
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )
    )
    vs_pseudobulk = _pseudobulk_vendi(adata, layer_key, _CONTROL_LABEL, n_pca_components, seed)
    pds_scores = _split_half_pds(adata, layer_key, _CONTROL_LABEL, seed)

    return {
        "P": P,
        "seed": seed,
        "vendi_score_cell": vs_cell,
        "vendi_score_pseudobulk": vs_pseudobulk,
        "pds_l1": pds_scores["pds_l1"],
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "G": G,
        "N0": N0,
        "Nk": Nk,
        "p_effect": p_effect,
        "effect_factor": effect_factor,
        "B": B,
    }


def run_effective_rank_sweep(
    *,
    n_trials: int = _DEFAULT_N_TRIALS,
    n_pca_components: int = 50,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Sample all DGP parameters (including P) per trial and score each."""
    inputs = load_parameter_estimation_inputs()
    rows: list[dict[str, object]] = []

    for trial in range(n_trials):
        seed = base_seed + trial
        rng = np.random.default_rng(seed)
        params = sample_parameters(PARAM_RANGES, rng)
        P = int(params["P"])
        G = int(params["G"])
        if P > G:
            print(f"  Skipping trial {trial}: P={P} > G={G}")
            continue
        print(f"  trial={trial}, P={P}, G={G}, seed={seed}")
        row = _generate_and_score(
            P=P,
            G=G,
            N0=int(params["N0"]),
            Nk=int(params["Nk"]),
            p_effect=float(params["p_effect"]),
            effect_factor=float(params["effect_factor"]),
            B=float(params["B"]),
            mu_l=float(params["mu_l"]),
            inputs=inputs,
            n_pca_components=n_pca_components,
            seed=seed,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_effective_rank(results: pd.DataFrame, output_dir: Path) -> None:
    """Save Vendi/PDS scaling plots and a cell-pseudobulk comparison."""
    apply_paper_plot_style()
    P = results["P"].to_numpy(dtype=float)
    p_effect, marker_areas, color_norm, size_handles = _scatter_encodings(results)

    # --- Figure A: Vendi cell + pseudobulk vs P ---
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), constrained_layout=True)
    for ax, col, label in [
        (axes[0], "vendi_score_cell", "Vendi (cell)"),
        (axes[1], "vendi_score_pseudobulk", "Vendi (pseudobulk)"),
    ]:
        y = results[col].to_numpy(dtype=float)
        finite = np.isfinite(P) & np.isfinite(y) & np.isfinite(p_effect) & (p_effect > 0)
        scatter = ax.scatter(
            P[finite],
            y[finite],
            c=p_effect[finite],
            s=marker_areas[finite],
            cmap="viridis",
            norm=color_norm,
            alpha=0.65,
        )
        lo = min(P[finite].min(), y[finite].min())
        hi = max(P[finite].max(), y[finite].max())
        ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1.5, label="y = x")
        if finite.sum() >= 3:
            r, p = pearsonr(P[finite], y[finite])
            slope, intercept = np.polyfit(P[finite], y[finite], 1)
            x_fit = np.linspace(P[finite].min(), P[finite].max(), 50)
            ax.plot(
                x_fit,
                slope * x_fit + intercept,
                "-",
                color="tab:red",
                linewidth=2.5,
                label=rf"OLS ($\beta_1 = {slope:.3f}$)",
            )
            ax.set_title(rf"Pearson $r = {r:.3f}$, $p = {p:.2e}$", fontsize=14)
        ax.set_xlabel("Number of perturbations (P)")
        ax.set_ylabel(label)
        _add_plot_guides(ax, size_handles)

    colorbar = fig.colorbar(scatter, ax=axes, pad=0.02)
    colorbar.set_label(r"$p_{\mathrm{effect}}$ (log scale)")
    fig.suptitle("DirectDGP: Vendi score vs P")
    vendi_path = output_dir / "effective_rank_vendi.png"
    fig.savefig(vendi_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Vendi plot to {vendi_path}")

    # --- Figure B: PDS-L1 vs P ---
    fig, ax = plt.subplots(figsize=(6.0, 5.8), constrained_layout=True)
    y = results["pds_l1"].to_numpy(dtype=float)
    finite = np.isfinite(P) & np.isfinite(y) & np.isfinite(p_effect) & (p_effect > 0)
    scatter = ax.scatter(
        P[finite],
        y[finite],
        c=p_effect[finite],
        s=marker_areas[finite],
        cmap="viridis",
        norm=color_norm,
        alpha=0.65,
    )
    if finite.sum() >= 3:
        r, p = pearsonr(P[finite], y[finite])
        slope, intercept = np.polyfit(P[finite], y[finite], 1)
        x_fit = np.linspace(P[finite].min(), P[finite].max(), 50)
        ax.plot(
            x_fit,
            slope * x_fit + intercept,
            "-",
            color="tab:red",
            linewidth=2.5,
            label=rf"OLS ($\beta_1 = {slope:.3f}$)",
        )
        ax.set_title(rf"Pearson $r = {r:.3f}$, $p = {p:.2e}$", fontsize=14)
    ax.set_xlabel("Number of perturbations (P)")
    ax.set_ylabel("PDS-L1")
    _add_plot_guides(ax, size_handles, position="lower-left")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"$p_{\mathrm{effect}}$ (log scale)")
    fig.suptitle("DirectDGP: PDS-L1 vs P")
    pds_path = output_dir / "effective_rank_pds.png"
    fig.savefig(pds_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PDS plot to {pds_path}")

    # --- Figure C: Cell vs pseudobulk Vendi ---
    fig, ax = plt.subplots(figsize=(6.0, 5.8), constrained_layout=True)
    vendi_cell = results["vendi_score_cell"].to_numpy(dtype=float)
    vendi_pseudobulk = results["vendi_score_pseudobulk"].to_numpy(dtype=float)
    finite = (
        np.isfinite(vendi_cell)
        & np.isfinite(vendi_pseudobulk)
        & np.isfinite(p_effect)
        & (p_effect > 0)
    )
    scatter = ax.scatter(
        vendi_cell[finite],
        vendi_pseudobulk[finite],
        c=p_effect[finite],
        s=marker_areas[finite],
        cmap="viridis",
        norm=color_norm,
        alpha=0.65,
    )
    lo = min(vendi_cell[finite].min(), vendi_pseudobulk[finite].min())
    hi = max(vendi_cell[finite].max(), vendi_pseudobulk[finite].max())
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1.5, label="y = x")
    if finite.sum() >= 3:
        rho, p = spearmanr(vendi_cell[finite], vendi_pseudobulk[finite])
        ax.set_title(rf"Spearman $\rho = {rho:.3f}$, $p = {p:.2e}$", fontsize=14)
    ax.set_xlabel("Vendi (cell)")
    ax.set_ylabel("Vendi (pseudobulk)")
    _add_plot_guides(ax, size_handles, position="below-identity")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"$p_{\mathrm{effect}}$ (log scale)")
    fig.suptitle("DirectDGP: Cell vs pseudobulk Vendi")
    comparison_path = output_dir / "effective_rank_cell_vs_pseudobulk.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cell-vs-pseudobulk plot to {comparison_path}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate Vendi effective-rank tracking with DirectDGP."
    )
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-trials", type=int, default=_DEFAULT_N_TRIALS)
    parser.add_argument("--n-pca-components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.n_trials} trials with all params sampled from PARAM_RANGES.")

    results = run_effective_rank_sweep(
        n_trials=args.n_trials,
        n_pca_components=args.n_pca_components,
        base_seed=args.seed,
    )

    csv_path = output_dir / "effective_rank_sweep.csv"
    results.to_csv(csv_path, index=False)
    print(f"Wrote results to {csv_path}")

    plot_effective_rank(results, output_dir)


if __name__ == "__main__":
    main()
