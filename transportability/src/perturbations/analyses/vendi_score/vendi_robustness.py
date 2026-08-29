r"""
Vendi diversity-score sensitivity to injected Gaussian noise.

Sweeps the perturbation-level Vendi score across injected Gaussian noise
levels, with repeated seeds, to characterize when the diversity-aware metric
is stable.

Example:
    python -m perturbations.analyses.vendi_score.vendi_robustness \\
        --name norman19 \\
        --dataset-path data/norman19/norman19_processed.h5ad \\
        --output-dir results/vendi_robustness
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from perturbations.analyses.common import NORM_LAYER_KEY
from perturbations.analyses.plot_utils import apply_paper_plot_style
from perturbations.analyses.util import ensure_normalized_log1p_layer, load_real_dataset
from perturbations.analyses.vendi_score.run_vendi import (
    _estimate_vendi_params,
    _split_half_pds,
)
from perturbations.metrics.reconstruction.vendi_score import (
    estimate_vendi_pseudobulk_sigma_squared,
    fit_vendi_pseudobulk_pca,
    vendi_score,
    vendi_score_pseudobulk,
)

matplotlib.use("Agg")

_DEFAULT_OUTPUT_DIR = "results/vendi_robustness"
_DEFAULT_N_PCA_COMPONENTS = 50
_DEFAULT_GAUSSIAN_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0)
_PLOT_METRICS = (
    ("vendi_cell_mean", "vendi_cell_std", "Vendi (cell)", "#3c5488"),
    ("vendi_pseudobulk_mean", "vendi_pseudobulk_std", "Vendi (pseudobulk)", "#e67700"),
)


def _parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    return [float(x) for x in text.split(",") if x.strip() != ""]


def _compute_vendi(
    adata: ad.AnnData,
    lognorm_layer: str,
    control_label: str,
    n_pca_components: int,
    seed: int,
    pca_model: object,
    gamma: float,
    outer_sigma_squared: float,
) -> float:
    """Compute the perturbation-level Vendi score for one dataset."""
    return float(
        vendi_score(
            ac=adata,
            n_pca_components=n_pca_components,
            layer_key=lognorm_layer,
            control_label=control_label,
            random_state=seed,
            gamma=gamma,
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )
    )


def _pseudobulk_matrix(
    adata: ad.AnnData,
    lognorm_layer: str,
    control_label: str,
) -> tuple[np.ndarray, int | None]:
    """Return ordered perturbation means and the optional control-row index."""
    labels = np.asarray(adata.obs["perturbation"])
    unique_labels = sorted(np.unique(labels).tolist())
    matrix = adata.layers[lognorm_layer]

    means: list[np.ndarray] = []
    control_idx: int | None = None
    for i, label in enumerate(unique_labels):
        mask = labels == label
        mean = matrix[mask, :].mean(axis=0)
        means.append(np.asarray(mean, dtype=np.float64).ravel())
        if control_label is not None and label == control_label:
            control_idx = i

    return np.vstack(means), control_idx


def _compute_vendi_pseudobulk(
    adata: ad.AnnData,
    lognorm_layer: str,
    control_label: str,
    pca_model: object,
    outer_sigma_squared: float,
) -> float:
    """Compute pseudobulk Vendi using a fixed observed reference."""
    pseudobulk, control_idx = _pseudobulk_matrix(adata, lognorm_layer, control_label)
    return float(
        vendi_score_pseudobulk(
            pseudobulk,
            control_idx=control_idx,
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )
    )


def _compute_scores(
    adata: ad.AnnData,
    lognorm_layer: str,
    control_label: str,
    n_pca_components: int,
    seed: int,
    pca_model: object,
    gamma: float,
    outer_sigma_squared: float,
    pseudobulk_pca_model: object | None,
    pseudobulk_sigma_squared: float | None,
) -> tuple[float, float, float]:
    """Compute cell/pseudobulk Vendi and PDS-L1, returning NaN on failure."""
    try:
        cell = _compute_vendi(
            adata,
            lognorm_layer,
            control_label,
            n_pca_components,
            seed,
            pca_model,
            gamma,
            outer_sigma_squared,
        )
    except (ValueError, KeyError):
        cell = float("nan")
    try:
        pseudobulk = _compute_vendi_pseudobulk(
            adata,
            lognorm_layer,
            control_label,
            pseudobulk_pca_model,
            pseudobulk_sigma_squared,
        )
    except (ValueError, KeyError):
        pseudobulk = float("nan")
    try:
        pds_l1 = _split_half_pds(adata, lognorm_layer, control_label, seed)["pds_l1"]
    except (ValueError, KeyError):
        pds_l1 = float("nan")
    return cell, pseudobulk, pds_l1


def _subsample_cells(adata: ad.AnnData, fraction: float, rng: np.random.Generator) -> ad.AnnData:
    """Return a random cell subsample at the given fraction."""
    if fraction >= 1.0:
        return adata
    n_keep = max(1, round(adata.n_obs * fraction))
    idx = rng.choice(adata.n_obs, size=n_keep, replace=False)
    return adata[np.sort(idx), :].copy()


def _inject_noise(
    adata: ad.AnnData,
    lognorm_layer: str,
    level: float,
    rng: np.random.Generator,
    control_label: str | None = None,
    noise_target: str = "all",
    gaussian_standard_noise: np.ndarray | None = None,
    gaussian_gene_std: np.ndarray | None = None,
    clip_gaussian_nonnegative: bool = True,
) -> ad.AnnData:
    """
    Return a copy of ``adata`` with zero-mean Gaussian noise added.

    Noise is scaled to each gene's own variability: the per-gene standard
    deviation is ``level * std_g``. ``level`` is therefore a dimensionless
    noise-to-signal ratio (alpha) that is comparable across datasets;
    ``alpha=1`` means the injected noise std equals each gene's signal std
    (i.e. half the total variance becomes noise).
    Supplying ``gaussian_standard_noise`` and ``gaussian_gene_std`` reuses one
    nested noise realization across levels. ``clip_gaussian_nonnegative`` should
    be disabled for signed latent feature spaces.

    ``noise_target="perturbed"`` restricts the noise to non-control cells
    (``obs['perturbation'] != control_label``); ``"all"`` noises every cell.
    """
    if level <= 0.0:
        return adata

    out = adata.copy()

    if noise_target == "perturbed":
        if control_label is None:
            raise ValueError("control_label is required when noise_target='perturbed'.")
        row_mask = np.asarray(out.obs["perturbation"]) != control_label
    else:
        row_mask = np.ones(out.n_obs, dtype=bool)

    matrix = out.layers[lognorm_layer]
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float64)
    # Scale noise to each gene's own variability: sigma_g = level * std_g,
    # so `level` is a dimensionless noise-to-signal ratio (alpha) comparable
    # across datasets. alpha=1 => noise std equals signal std per gene.
    gene_std = (
        dense.std(axis=0, keepdims=True)
        if gaussian_gene_std is None
        else np.asarray(gaussian_gene_std, dtype=np.float64)
    )
    standard_noise = (
        rng.standard_normal(size=dense.shape)
        if gaussian_standard_noise is None
        else np.asarray(gaussian_standard_noise, dtype=np.float64)
    )
    if standard_noise.shape != dense.shape:
        raise ValueError(
            "gaussian_standard_noise must match the selected layer shape. "
            + f"Got {standard_noise.shape} and {dense.shape}."
        )
    noise = standard_noise * (level * gene_std)
    noise[~row_mask, :] = 0.0
    dense = dense + noise
    if clip_gaussian_nonnegative:
        np.maximum(dense, 0.0, out=dense)
    out.layers[lognorm_layer] = dense.astype(np.float32)

    return out


def run_sensitivity(
    adata: ad.AnnData,
    name: str,
    lognorm_layer: str = NORM_LAYER_KEY,
    control_label: str = "control",
    noise_levels: list[float] | None = None,
    n_seeds: int = 10,
    n_pca_components: int = _DEFAULT_N_PCA_COMPONENTS,
    vendi_max_cells: int | None = None,
    seed: int = 0,
    noise_target: str = "perturbed",
    clip_gaussian_nonnegative: bool = True,
) -> pd.DataFrame:
    """
    Sweep the Vendi score across Gaussian noise levels, returning a long results table.

    Args:
        adata: Dataset with a log-normalized ``lognorm_layer`` and
            ``obs['perturbation']``.
        name: Dataset label recorded in the output.
        lognorm_layer: Log-normalized layer used by the Vendi score.
        control_label: Control label in ``obs['perturbation']``.
        noise_levels: Gaussian noise-to-signal ratios (alpha) for the noise sweep.
        n_seeds: Repeats per condition.
        n_pca_components: PCA components for the Vendi embedding.
        vendi_max_cells: Optional cap on cells used in the noise sweep. ``None``
            uses all cells.
        seed: Base RNG seed.
        noise_target: ``"all"`` noises every cell; ``"perturbed"`` restricts
            noise to non-control cells.
        clip_gaussian_nonnegative: Clip Gaussian-noised values at zero for
            nonnegative expression spaces. Disable for signed latent features.

    Returns:
        Long DataFrame with columns
        ``["dataset", "sweep", "value", "noise_type", "noise_variance_fraction",
        "seed", "vendi_cell", "vendi_pseudobulk", "pds_l1"]``. ``value`` is the
        noise-to-signal ratio (alpha, per-gene) and
        ``noise_variance_fraction = alpha^2 / (1 + alpha^2)``.
    """
    noise_levels = noise_levels if noise_levels is not None else list(_DEFAULT_GAUSSIAN_LEVELS)
    rows: list[dict[str, object]] = []

    # Noise sweep: fixed, optionally capped size; inject noise and recompute Vendi.
    base = adata
    if vendi_max_cells is not None:
        base = _subsample_cells(
            adata,
            min(1.0, vendi_max_cells / max(1, adata.n_obs)),
            np.random.default_rng(seed),
        )
    base_layer = base.layers[lognorm_layer]
    base_dense = (
        base_layer.toarray()
        if sparse.issparse(base_layer)
        else np.asarray(base_layer, dtype=np.float64)
    )
    gaussian_gene_std = base_dense.std(axis=0, keepdims=True)
    gaussian_noise_by_seed = {
        s: np.random.default_rng(seed + s).standard_normal(size=base_dense.shape)
        for s in range(n_seeds)
    }
    total_iterations = len(noise_levels) * n_seeds
    iteration = 0
    for level in noise_levels:
        # `level` is alpha (noise-to-signal std ratio); the fraction of total
        # variance that is noise is alpha^2 / (1 + alpha^2).
        variance_fraction = (level**2) / (1.0 + level**2)
        for s in range(n_seeds):
            iteration += 1
            # Liveness signal: the sweep otherwise prints nothing until it finishes.
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name}: noise sweep "
                f"{iteration}/{total_iterations} (alpha={level:g}, seed={s})",
                flush=True,
            )
            rng = np.random.default_rng(seed + s)
            noised = _inject_noise(
                base,
                lognorm_layer,
                level,
                rng,
                control_label=control_label,
                noise_target=noise_target,
                gaussian_standard_noise=gaussian_noise_by_seed[s],
                gaussian_gene_std=gaussian_gene_std,
                clip_gaussian_nonnegative=clip_gaussian_nonnegative,
            )

            # Re-estimate Vendi parameters from the noised data itself for every
            # condition, so no iteration reuses another iteration's calibration.
            pca_model, gamma, outer_sigma_squared = _estimate_vendi_params(
                noised, lognorm_layer, control_label, n_pca_components, seed + s
            )
            observed_pseudobulk, observed_control_idx = _pseudobulk_matrix(
                noised, lognorm_layer, control_label
            )
            n_observed_perturbations = observed_pseudobulk.shape[0] - int(
                observed_control_idx is not None
            )
            pseudobulk_pca_model = None
            pseudobulk_sigma_squared = None
            if n_observed_perturbations > 1:
                pseudobulk_pca_model = fit_vendi_pseudobulk_pca(
                    observed_pseudobulk,
                    control_idx=observed_control_idx,
                    n_pca_components=n_pca_components,
                    random_state=seed + s,
                )
                pseudobulk_sigma_squared = estimate_vendi_pseudobulk_sigma_squared(
                    ac=noised,
                    pca_model=pseudobulk_pca_model,
                    layer_key=lognorm_layer,
                    control_label=control_label,
                    random_state=seed + s,
                )

            vendi_cell, vendi_pseudobulk, pds_l1 = _compute_scores(
                noised,
                lognorm_layer,
                control_label,
                n_pca_components,
                seed + s,
                pca_model,
                gamma,
                outer_sigma_squared,
                pseudobulk_pca_model,
                pseudobulk_sigma_squared,
            )
            rows.append(
                {
                    "dataset": name,
                    "sweep": "noise",
                    "value": level,
                    "noise_type": "gaussian",
                    "noise_variance_fraction": variance_fraction,
                    "seed": s,
                    "vendi_cell": vendi_cell,
                    "vendi_pseudobulk": vendi_pseudobulk,
                    "pds_l1": pds_l1,
                }
            )

    return pd.DataFrame(rows)


def summarize_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the Vendi sweep into mean/std/CV per condition across seeds."""
    grouped = results.groupby(["dataset", "sweep", "value", "noise_type"], as_index=False).agg(
        vendi_cell_mean=("vendi_cell", "mean"),
        vendi_cell_std=("vendi_cell", "std"),
        vendi_pseudobulk_mean=("vendi_pseudobulk", "mean"),
        vendi_pseudobulk_std=("vendi_pseudobulk", "std"),
        pds_l1_mean=("pds_l1", "mean"),
        pds_l1_std=("pds_l1", "std"),
        noise_variance_fraction=("noise_variance_fraction", "first"),
        n_seeds=("seed", "count"),
    )
    grouped["vendi_cell_cv"] = grouped["vendi_cell_std"] / grouped["vendi_cell_mean"].abs()
    grouped["vendi_pseudobulk_cv"] = (
        grouped["vendi_pseudobulk_std"] / grouped["vendi_pseudobulk_mean"].abs()
    )
    grouped["pds_l1_cv"] = grouped["pds_l1_std"] / grouped["pds_l1_mean"].abs()
    return grouped


def plot_robustness(summary: pd.DataFrame, output_dir: Path, name: str) -> None:
    """Save dataset-size and noise robustness plots from a sensitivity summary."""
    apply_paper_plot_style()

    sweep_specs = (("noise", "Noise level", "Injected noise"),)
    for sweep, default_x_label, sweep_title in sweep_specs:
        sweep_data = summary.loc[summary["sweep"] == sweep].sort_values("value")
        if sweep_data.empty:
            continue

        x_column = "value"
        x_label = default_x_label
        if sweep == "noise":
            x_column = "noise_variance_fraction"
            x_label = r"Noise variance fraction ($\alpha^2 / (1 + \alpha^2)$)"

        x = sweep_data[x_column].to_numpy(dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), constrained_layout=True)
        for ax, (mean_column, std_column, metric_label, color) in zip(
            axes, _PLOT_METRICS, strict=True
        ):
            mean = sweep_data[mean_column].to_numpy(dtype=float)
            std = np.nan_to_num(sweep_data[std_column].to_numpy(dtype=float), nan=0.0)
            finite = np.isfinite(x) & np.isfinite(mean) & np.isfinite(std)
            if not finite.any():
                continue

            ax.plot(
                x[finite],
                mean[finite],
                "-o",
                color=color,
                markersize=8,
                label="Mean",
            )
            ax.fill_between(
                x[finite],
                mean[finite] - std[finite],
                mean[finite] + std[finite],
                color=color,
                alpha=0.2,
                label=r"$\pm$ 1 SD",
            )
            ax.set_title(metric_label, fontsize=14)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Vendi score")
            ax.set_axisbelow(True)
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
            ax.legend()

        fig.suptitle(f"{name}: Vendi robustness to {sweep_title.lower()}")
        output_path = output_dir / f"{name}_vendi_robustness_{sweep}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {sweep_title.lower()} plot to {output_path}")

        pds_mean = sweep_data["pds_l1_mean"].to_numpy(dtype=float)
        pds_std = np.nan_to_num(sweep_data["pds_l1_std"].to_numpy(dtype=float), nan=0.0)
        finite = np.isfinite(x) & np.isfinite(pds_mean) & np.isfinite(pds_std)
        if not finite.any():
            continue

        fig, ax = plt.subplots(figsize=(6.0, 5.8), constrained_layout=True)
        ax.plot(
            x[finite],
            pds_mean[finite],
            "-o",
            color="#c92a2a",
            markersize=8,
            label="Mean",
        )
        ax.fill_between(
            x[finite],
            pds_mean[finite] - pds_std[finite],
            pds_mean[finite] + pds_std[finite],
            color="#c92a2a",
            alpha=0.2,
            label=r"$\pm$ 1 SD",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("PDS-L1")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.legend()
        fig.suptitle(f"{name}: PDS-L1 robustness to {sweep_title.lower()}")
        pds_path = output_dir / f"{name}_pds_l1_robustness_{sweep}.png"
        fig.savefig(pds_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PDS-L1 {sweep_title.lower()} plot to {pds_path}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Vendi score sensitivity to Gaussian noise.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--dataset-path", required=True, help="Path to the real dataset .h5ad.")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--lognorm-layer", default=NORM_LAYER_KEY)
    parser.add_argument(
        "--noise-target",
        default="perturbed",
        choices=["all", "perturbed"],
        help="Inject noise into all cells or only non-control (perturbed) cells.",
    )
    parser.add_argument(
        "--noise-levels",
        default=None,
        help="Comma-separated Gaussian noise-to-signal ratios (alpha); defaults to a preset grid.",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-pca-components", type=int, default=_DEFAULT_N_PCA_COMPONENTS)
    parser.add_argument(
        "--vendi-max-cells",
        type=int,
        default=None,
        help="Optionally cap noise-sweep cells; defaults to all cells.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    """Parse arguments, load/generate a dataset, and write the sensitivity sweep."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or Path(args.dataset_path).stem
    print(f"Loading real dataset {args.dataset_path} ...")
    adata, _ = load_real_dataset(dataset_path=args.dataset_path)

    ensure_normalized_log1p_layer(
        adata, output_layer_key=args.lognorm_layer, source_layer=args.counts_layer
    )
    print(f"Dataset {name!r}: {adata.n_obs} cells x {adata.n_vars} genes.")

    results = run_sensitivity(
        adata,
        name=name,
        lognorm_layer=args.lognorm_layer,
        control_label=args.control_label,
        noise_levels=(
            _parse_float_list(args.noise_levels) if args.noise_levels is not None else None
        ),
        n_seeds=args.n_seeds,
        n_pca_components=args.n_pca_components,
        vendi_max_cells=args.vendi_max_cells,
        seed=args.seed,
        noise_target=args.noise_target,
    )
    summary = summarize_sensitivity(results)

    results_path = output_dir / f"{name}_vendi_robustness.csv"
    summary_path = output_dir / f"{name}_vendi_robustness_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote raw results to {results_path}")
    print(f"Wrote summary to {summary_path}")
    plot_robustness(summary, output_dir, name)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
