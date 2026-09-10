r"""
Test perturbation-diversity metrics under DirectDGP null and signal conditions.

The experiment holds the number of perturbations fixed and sweeps perturbation
effect strength. ``effect_factor=1`` is the population null because every
perturbation has the same mean response. Repeating the sweep at multiple
``Nk`` values tests sensitivity to pseudobulk sampling noise.

Example:
    python -m perturbations.analyses.vendi_score.vendi_null_signal \
        --output-dir results/vendi_null_signal \
        --repeats 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perturbations.analyses.common import NORM_LAYER_KEY
from perturbations.analyses.plot_utils import apply_paper_plot_style
from perturbations.analyses.util import compute_means_by_perturbation
from perturbations.analyses.vendi_score.run_vendi import _pseudobulk_vendi
from perturbations.data.dgp.directDGP import directDGP
from perturbations.metrics.reconstruction.vendi_score import covariance_effective_rank

from ..synthetic_simulations.sampling import load_parameter_estimation_inputs

matplotlib.use("Agg")

_DEFAULT_OUTPUT_DIR = "results/vendi_null_signal"
_DEFAULT_EFFECT_FACTORS = (1.0, 1.2, 2.0, 5.0)
_DEFAULT_NK_VALUES = (64, 256)
_DEFAULT_REPEATS = 5
_DEFAULT_P = 20
_DEFAULT_G = 1024
_DEFAULT_N0 = 512
_DEFAULT_P_EFFECT = 0.05
_DEFAULT_B = 1.0
_DEFAULT_MU_L = 2.5
_CONTROL_LABEL = "control"


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    """Parse a comma-separated tuple of floats."""
    return tuple(float(value) for value in text.split(",") if value.strip())


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    """Parse a comma-separated tuple of integers."""
    return tuple(int(value) for value in text.split(",") if value.strip())


def _generate_and_score_pseudobulk(
    *,
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
    """Generate one DirectDGP dataset and compute both pseudobulk metrics."""
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

    vendi = _pseudobulk_vendi(
        adata,
        NORM_LAYER_KEY,
        _CONTROL_LABEL,
        n_pca_components,
        seed,
    )
    labels = np.asarray(adata.obs["perturbation"])
    perturbation_ids = np.unique(labels[labels != _CONTROL_LABEL])
    pseudobulk = compute_means_by_perturbation(
        adata_view=adata,
        perturbation_ids=perturbation_ids,
        layer_key=NORM_LAYER_KEY,
    )
    effective_rank = covariance_effective_rank(pseudobulk)

    return {
        "P": P,
        "G": G,
        "N0": N0,
        "Nk": Nk,
        "p_effect": p_effect,
        "effect_factor": effect_factor,
        "B": B,
        "mu_l": mu_l,
        "seed": seed,
        "vendi_score_pseudobulk": vendi,
        "effective_rank_pseudobulk": effective_rank,
    }


def run_null_signal_sweep(
    *,
    repeats: int = _DEFAULT_REPEATS,
    n_pca_components: int = 50,
    base_seed: int = 0,
    P: int = _DEFAULT_P,
    G: int = _DEFAULT_G,
    N0: int = _DEFAULT_N0,
    Nk_values: tuple[int, ...] = _DEFAULT_NK_VALUES,
    p_effect: float = _DEFAULT_P_EFFECT,
    effect_factors: tuple[float, ...] = _DEFAULT_EFFECT_FACTORS,
    B: float = _DEFAULT_B,
    mu_l: float = _DEFAULT_MU_L,
) -> pd.DataFrame:
    """Run a paired DirectDGP null-versus-signal sweep at fixed ``P``."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")
    if P < 2 or G < P:
        raise ValueError(f"Expected 2 <= P <= G, got P={P}, G={G}.")
    if not 0.0 <= p_effect <= 1.0:
        raise ValueError(f"p_effect must be between 0 and 1, got {p_effect}.")
    if not Nk_values or any(Nk < 1 for Nk in Nk_values):
        raise ValueError("Nk_values must contain positive integers.")
    if N0 < 2 * max(Nk_values):
        raise ValueError("N0 must be at least twice the largest Nk for matched null splits.")
    if not effect_factors or any(factor < 1.0 for factor in effect_factors):
        raise ValueError("effect_factors must contain values >= 1.")
    if 1.0 not in effect_factors:
        raise ValueError("effect_factors must include the population-null value 1.0.")

    inputs = load_parameter_estimation_inputs()
    rows: list[dict[str, object]] = []
    total = len(Nk_values) * repeats * len(effect_factors)
    completed = 0
    for Nk in Nk_values:
        for repeat in range(repeats):
            seed = base_seed + repeat
            for effect_factor in effect_factors:
                completed += 1
                print(
                    f"DirectDGP null-signal {completed}/{total}: "
                    f"repeat={repeat}, P={P}, Nk={Nk}, "
                    f"effect_factor={effect_factor:g}, seed={seed}",
                    flush=True,
                )
                row = _generate_and_score_pseudobulk(
                    P=P,
                    G=G,
                    N0=N0,
                    Nk=Nk,
                    p_effect=p_effect,
                    effect_factor=effect_factor,
                    B=B,
                    mu_l=mu_l,
                    inputs=inputs,
                    n_pca_components=n_pca_components,
                    seed=seed,
                )
                row.update(
                    {
                        "repeat": repeat,
                        "is_population_null": effect_factor == 1.0,
                        "expected_null_diversity": (1.0 if effect_factor == 1.0 else float("nan")),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_null_signal_sweep(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize each effect-strength and sample-size condition across repeats."""
    return results.groupby(["P", "Nk", "p_effect", "effect_factor"], as_index=False).agg(
        vendi_score_pseudobulk_mean=("vendi_score_pseudobulk", "mean"),
        vendi_score_pseudobulk_std=("vendi_score_pseudobulk", "std"),
        effective_rank_pseudobulk_mean=("effective_rank_pseudobulk", "mean"),
        effective_rank_pseudobulk_std=("effective_rank_pseudobulk", "std"),
        n_repeats=("seed", "count"),
        is_population_null=("is_population_null", "first"),
        expected_null_diversity=("expected_null_diversity", "first"),
    )


def plot_null_signal_sweep(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot pseudobulk Vendi and effective rank against perturbation strength."""
    apply_paper_plot_style()
    metrics = (
        (
            "vendi_score_pseudobulk_mean",
            "vendi_score_pseudobulk_std",
            "Vendi",
            "#e67700",
            4,
        ),
        (
            "effective_rank_pseudobulk_mean",
            "effective_rank_pseudobulk_std",
            "Effective rank",
            "#2b8a3e",
            3,
        ),
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    for mean_column, std_column, label, color, line_zorder in metrics:
        for line_idx, (Nk, group) in enumerate(summary.groupby("Nk", sort=True)):
            group = group.sort_values("effect_factor")
            x = group["effect_factor"].to_numpy(dtype=float)
            mean = group[mean_column].to_numpy(dtype=float)
            std = np.nan_to_num(group[std_column].to_numpy(dtype=float), nan=0.0)
            linestyle = "-" if line_idx == 0 else "--"
            ax.plot(
                x,
                mean,
                marker="o",
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=rf"{label}, $N_P={int(Nk)}$",
                zorder=line_zorder,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15,
                zorder=line_zorder - 2,
            )
    P = int(summary["P"].iloc[0])
    ax.axhline(1.0, linestyle=":", color="grey", linewidth=1.5, label="Null target")
    ax.axhline(P, linestyle=":", color="grey", linewidth=1.5, label=rf"$P = {P}$")
    ax.set_xlabel(r"Effect factor ($\epsilon$)")
    ax.set_ylabel("Effective diversity")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(loc="lower right", ncol=1, fontsize=13)

    ax.set_ylim(0.8, P + 0.5)
    output_path = output_dir / "vendi_null_signal.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved null-signal plot to {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare pseudobulk Vendi and effective rank under DirectDGP null and signal."
    )
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    parser.add_argument("--n-pca-components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--P", type=int, default=_DEFAULT_P)
    parser.add_argument("--G", type=int, default=_DEFAULT_G)
    parser.add_argument("--N0", type=int, default=_DEFAULT_N0)
    parser.add_argument("--p-effect", type=float, default=_DEFAULT_P_EFFECT)
    parser.add_argument(
        "--Nk-values",
        type=_parse_int_tuple,
        default=_DEFAULT_NK_VALUES,
        help="Comma-separated perturbed-cell counts.",
    )
    parser.add_argument(
        "--effect-factors",
        type=_parse_float_tuple,
        default=_DEFAULT_EFFECT_FACTORS,
        help="Comma-separated factors; must include the null value 1.",
    )
    return parser


def main() -> None:
    """Run the controlled sweep and write raw, summarized, and plotted results."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_null_signal_sweep(
        repeats=args.repeats,
        n_pca_components=args.n_pca_components,
        base_seed=args.seed,
        P=args.P,
        G=args.G,
        N0=args.N0,
        Nk_values=args.Nk_values,
        p_effect=args.p_effect,
        effect_factors=args.effect_factors,
    )
    summary = summarize_null_signal_sweep(results)

    results_path = output_dir / "vendi_null_signal.csv"
    summary_path = output_dir / "vendi_null_signal_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote raw results to {results_path}")
    print(f"Wrote summary to {summary_path}")
    plot_null_signal_sweep(summary, output_dir)


if __name__ == "__main__":
    main()
