"""
Stress-test pseudobulk Vendi calibration with DirectDGP cell pools.

The normalized population mean is not analytic under DirectDGP, so a large,
disjoint cell pool approximates the deterministic population-mean oracle.
"""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from perturbations.analyses.common import NORM_LAYER_KEY
from perturbations.analyses.plot_utils import apply_paper_plot_style
from perturbations.analyses.synthetic_simulations.sampling import (
    load_parameter_estimation_inputs,
)
from perturbations.analyses.util import compute_means_by_perturbation
from perturbations.data.dgp.directDGP import directDGP
from perturbations.metrics.reconstruction.distance_util import pairwise_squared_distances
from perturbations.metrics.reconstruction.vendi_score import (
    estimate_vendi_pseudobulk_sigma_squared,
    fit_vendi_pseudobulk_pca,
    vendi_score_pseudobulk,
)

matplotlib.use("Agg")

_CONTROL_LABEL = "control"
_LAYER_KEY = NORM_LAYER_KEY
_DEFAULT_OUTPUT_DIR = "results/vendi_sample_size_bias"
_DEFAULT_MINIMUM_SAMPLE_SIZE = 64
_DEFAULT_SAMPLE_SIZES = (64, 128, 256)
_DEFAULT_REPEATS = 10
_DEFAULT_N_GENES = 1024
_DEFAULT_N_CONTROLS = 512
_DEFAULT_P_EFFECT = 0.1
_DEFAULT_EFFECT_FACTORS = (1.0, 1.5, 2.0, 5.0)
_DEFAULT_B = 0.0
_DEFAULT_MU_L = 2.5
_DEFAULT_ORACLE_POOL_SIZE = 2048
_DESIGN_ORDER = ("balanced", "moderate", "severe")


def sample_size_designs(
    n_perturbations: int,
    reference_size: int,
    minimum_sample_size: int = _DEFAULT_MINIMUM_SAMPLE_SIZE,
) -> dict[str, np.ndarray]:
    """Construct threshold-compliant vectors with a common median and total count."""
    if n_perturbations < 4 or n_perturbations % 4 != 0:
        raise ValueError("n_perturbations must be a positive multiple of four.")
    if reference_size < 1:
        raise ValueError("reference_size must be positive.")
    if minimum_sample_size < 1:
        raise ValueError("minimum_sample_size must be positive.")
    if reference_size - minimum_sample_size < 2:
        raise ValueError("reference_size must exceed minimum_sample_size by at least two cells.")

    quarter = n_perturbations // 4
    half = n_perturbations // 2
    moderate_low = (reference_size + minimum_sample_size) // 2
    severe_low = minimum_sample_size
    moderate_high = 2 * reference_size - moderate_low
    severe_high = 2 * reference_size - severe_low
    return {
        "balanced": np.full(n_perturbations, reference_size, dtype=np.int64),
        "moderate": np.concatenate(
            [
                np.full(quarter, moderate_low, dtype=np.int64),
                np.full(half, reference_size, dtype=np.int64),
                np.full(quarter, moderate_high, dtype=np.int64),
            ]
        ),
        "severe": np.concatenate(
            [
                np.full(quarter, severe_low, dtype=np.int64),
                np.full(half, reference_size, dtype=np.int64),
                np.full(quarter, severe_high, dtype=np.int64),
            ]
        ),
    }


def _validate_experiment_dimensions(
    n_perturbations: int,
    n_genes: int,
    n_controls: int,
    required_split_size: int,
) -> None:
    """Validate dimensions needed for DirectDGP and disjoint control splits."""
    if n_perturbations < 2:
        raise ValueError("n_perturbations must be at least two.")
    if n_genes < n_perturbations:
        raise ValueError("n_genes must be at least n_perturbations for DirectDGP.")
    if n_controls < 2 * required_split_size:
        raise ValueError(
            "n_controls must be at least twice the largest required calibration split size."
        )


def _generate_directdgp_pool(
    *,
    inputs: dict[str, np.ndarray],
    n_genes: int,
    n_controls: int,
    cells_per_perturbation: int,
    n_perturbations: int,
    p_effect: float,
    effect_factor: float,
    B: float,
    mu_l: float,
    seed: int,
) -> ad.AnnData:
    """Generate one normalized DirectDGP pool with balanced perturbation sizes."""
    adata, _ = directDGP(
        G=n_genes,
        N0=n_controls,
        Nk=cells_per_perturbation,
        P=n_perturbations,
        p_effect=p_effect,
        effect_factor=effect_factor,
        B=B,
        mu_l=mu_l,
        all_theta=inputs["all_theta"],
        control_mu=inputs["control_mu"],
        pert_mu=inputs["pert_mu"],
        gene_names=inputs["gene_names"],
        control_label=_CONTROL_LABEL,
        seed=seed,
        normalize=True,
        normalized_layer_key=_LAYER_KEY,
    )
    return adata


def _resolve_dgp_inputs(
    dgp_inputs: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Load the fitted DirectDGP parameters shared with the null-signal analysis."""
    if dgp_inputs is not None:
        return dgp_inputs
    try:
        return load_parameter_estimation_inputs()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "DirectDGP fitted parameters are required. Generate the files under "
            + "results/synthetic_simulations/parameter_estimation used by "
            + "vendi_null_signal before running this analysis."
        ) from error


def _ordered_perturbation_labels(adata: ad.AnnData) -> np.ndarray:
    """Return DirectDGP perturbation labels ordered by integer perturbation ID."""
    obs = cast(pd.DataFrame, adata.obs)
    pairs = cast(
        pd.DataFrame,
        obs.loc[
            obs["perturbation"] != _CONTROL_LABEL,
            ["perturbation", "perturbation_id"],
        ]
        .drop_duplicates()
        .sort_values("perturbation_id"),
    )
    return pairs["perturbation"].to_numpy(dtype=object)


def _select_directdgp_pool(
    adata: ad.AnnData,
    perturbation_labels: np.ndarray,
    sample_sizes: np.ndarray,
    *,
    offset: int = 0,
    include_controls: bool,
) -> ad.AnnData:
    """Select one disjoint interval from each perturbation group."""
    sizes = np.asarray(sample_sizes, dtype=np.int64)
    if sizes.shape != (perturbation_labels.size,):
        raise ValueError(
            "sample_sizes must contain one value per perturbation. "
            + f"Got {sizes.size} sizes for {perturbation_labels.size} perturbations."
        )
    if offset < 0 or np.any(sizes <= 0):
        raise ValueError("offset must be nonnegative and sample sizes must be positive.")

    observed_labels = np.asarray(adata.obs["perturbation"])
    selected_parts: list[np.ndarray] = []
    if include_controls:
        selected_parts.append(np.flatnonzero(observed_labels == _CONTROL_LABEL))
    for perturbation_label, sample_size in zip(
        perturbation_labels,
        sizes,
        strict=True,
    ):
        group_indices = np.flatnonzero(observed_labels == perturbation_label)
        stop = offset + int(sample_size)
        if stop > group_indices.size:
            raise ValueError(
                f"Requested rows [{offset}:{stop}] for perturbation "
                + f"{perturbation_label!r}, but its pool contains {group_indices.size} cells."
            )
        selected_parts.append(group_indices[offset:stop])

    selected_indices = np.concatenate(selected_parts)
    return adata[selected_indices, :].copy()


def _pseudobulk_from_adata(
    adata: ad.AnnData,
    perturbation_labels: np.ndarray,
) -> np.ndarray:
    """Compute normalized DirectDGP means in stable perturbation-ID order."""
    return compute_means_by_perturbation(
        adata_view=adata,
        perturbation_ids=perturbation_labels,
        layer_key=_LAYER_KEY,
    )


def _fit_and_score_observed(
    adata: ad.AnnData,
    observed_pseudobulk: np.ndarray,
    n_splits: int,
    null_quantile: float,
    target_null_similarity: float,
    random_state: int,
) -> tuple[PCA, float, float]:
    """Fit observed-reference PCA and null scale, then score observed pseudobulks."""
    pca_model = fit_vendi_pseudobulk_pca(
        observed_pseudobulk,
        n_pca_components=observed_pseudobulk.shape[1],
        random_state=random_state,
    )
    sigma_squared = estimate_vendi_pseudobulk_sigma_squared(
        ac=adata,
        pca_model=pca_model,
        layer_key=_LAYER_KEY,
        control_label=_CONTROL_LABEL,
        n_splits=n_splits,
        null_quantile=null_quantile,
        target_null_similarity=target_null_similarity,
        random_state=random_state,
    )
    score = vendi_score_pseudobulk(
        observed_pseudobulk,
        pca_model=pca_model,
        outer_sigma_squared=sigma_squared,
    )
    return pca_model, sigma_squared, score


def _mean_off_diagonal_similarity(
    pseudobulk: np.ndarray,
    pca_model: PCA,
    sigma_squared: float,
) -> float:
    """Return the mean RBF similarity between distinct pseudobulk rows."""
    embedded = np.asarray(pca_model.transform(pseudobulk), dtype=np.float64)
    distances_squared = pairwise_squared_distances(embedded)
    similarities = np.exp(-distances_squared / (2.0 * sigma_squared))
    upper_triangle = np.triu_indices(pseudobulk.shape[0], k=1)
    return float(np.mean(similarities[upper_triangle]))


def run_heterogeneous_null_experiment(
    *,
    repeats: int = _DEFAULT_REPEATS,
    n_perturbations: int = 20,
    n_genes: int = _DEFAULT_N_GENES,
    n_controls: int = _DEFAULT_N_CONTROLS,
    reference_size: int = 128,
    minimum_sample_size: int = _DEFAULT_MINIMUM_SAMPLE_SIZE,
    n_splits: int = 200,
    null_quantile: float = 0.95,
    target_null_similarity: float = 0.95,
    p_effect: float = _DEFAULT_P_EFFECT,
    effect_factors: tuple[float, ...] = _DEFAULT_EFFECT_FACTORS,
    B: float = _DEFAULT_B,
    mu_l: float = _DEFAULT_MU_L,
    base_seed: int = 0,
    dgp_inputs: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Sweep DirectDGP effect strength and perturbation-count heterogeneity."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if n_splits < 1:
        raise ValueError("n_splits must be positive.")
    if not 0.0 <= p_effect <= 1.0:
        raise ValueError("p_effect must be between zero and one.")
    if not effect_factors or any(factor < 1.0 for factor in effect_factors):
        raise ValueError("effect_factors must contain values greater than or equal to one.")

    designs = sample_size_designs(
        n_perturbations,
        reference_size,
        minimum_sample_size,
    )
    _validate_experiment_dimensions(
        n_perturbations,
        n_genes,
        n_controls,
        reference_size,
    )
    max_count = max(int(counts.max()) for counts in designs.values())
    inputs = _resolve_dgp_inputs(dgp_inputs)
    rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        seed = base_seed + repeat
        for effect_factor in effect_factors:
            pool = _generate_directdgp_pool(
                inputs=inputs,
                n_genes=n_genes,
                n_controls=n_controls,
                cells_per_perturbation=max_count,
                n_perturbations=n_perturbations,
                p_effect=p_effect,
                effect_factor=effect_factor,
                B=B,
                mu_l=mu_l,
                seed=seed,
            )
            perturbation_labels = _ordered_perturbation_labels(pool)

            for design_name in _DESIGN_ORDER:
                sample_sizes = designs[design_name]
                adata = _select_directdgp_pool(
                    pool,
                    perturbation_labels,
                    sample_sizes,
                    include_controls=True,
                )
                observed_pseudobulk = _pseudobulk_from_adata(adata, perturbation_labels)
                pca_model, sigma_squared, score = _fit_and_score_observed(
                    adata,
                    observed_pseudobulk,
                    n_splits,
                    null_quantile,
                    target_null_similarity,
                    seed,
                )
                split_size = min(int(np.median(sample_sizes)), n_controls // 2)
                rows.append(
                    {
                        "experiment": "heterogeneous_null",
                        "design": design_name,
                        "repeat": repeat,
                        "seed": seed,
                        "n_perturbations": n_perturbations,
                        "n_genes": n_genes,
                        "n_controls": n_controls,
                        "minimum_sample_size": minimum_sample_size,
                        "n_min": int(sample_sizes.min()),
                        "n_median": float(np.median(sample_sizes)),
                        "n_max": int(sample_sizes.max()),
                        "n_cv": float(np.std(sample_sizes) / np.mean(sample_sizes)),
                        "n_null": split_size,
                        "n_splits": n_splits,
                        "null_quantile": null_quantile,
                        "target_null_similarity": target_null_similarity,
                        "p_effect": p_effect,
                        "effect_factor": effect_factor,
                        "B": B,
                        "mu_l": mu_l,
                        "vendi_observed": score,
                        "mean_off_diagonal_similarity": _mean_off_diagonal_similarity(
                            observed_pseudobulk,
                            pca_model,
                            sigma_squared,
                        ),
                        "sigma_null_squared": sigma_squared,
                    }
                )

    return pd.DataFrame(rows)


def run_deterministic_oracle_experiment(
    *,
    repeats: int = _DEFAULT_REPEATS,
    sample_sizes: tuple[int, ...] = _DEFAULT_SAMPLE_SIZES,
    n_perturbations: int = 20,
    n_genes: int = _DEFAULT_N_GENES,
    n_controls: int = _DEFAULT_N_CONTROLS,
    minimum_sample_size: int = _DEFAULT_MINIMUM_SAMPLE_SIZE,
    oracle_pool_size: int = _DEFAULT_ORACLE_POOL_SIZE,
    n_splits: int = 200,
    null_quantile: float = 0.95,
    target_null_similarity: float = 0.95,
    p_effect: float = _DEFAULT_P_EFFECT,
    effect_factors: tuple[float, ...] = _DEFAULT_EFFECT_FACTORS,
    B: float = _DEFAULT_B,
    mu_l: float = _DEFAULT_MU_L,
    base_seed: int = 0,
    dgp_inputs: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Compare large-pool and sampling-matched DirectDGP mean oracles."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if minimum_sample_size < 1:
        raise ValueError("minimum_sample_size must be positive.")
    if not sample_sizes or any(sample_size < minimum_sample_size for sample_size in sample_sizes):
        raise ValueError(
            "sample_sizes must contain integers greater than or equal to "
            + f"minimum_sample_size={minimum_sample_size}."
        )
    if oracle_pool_size <= max(sample_sizes):
        raise ValueError("oracle_pool_size must exceed the largest target sample size.")
    if n_splits < 1:
        raise ValueError("n_splits must be positive.")
    if not 0.0 <= p_effect <= 1.0:
        raise ValueError("p_effect must be between zero and one.")
    if not effect_factors or any(factor < 1.0 for factor in effect_factors):
        raise ValueError("effect_factors must contain values greater than or equal to one.")

    max_count = max(sample_sizes)
    _validate_experiment_dimensions(
        n_perturbations,
        n_genes,
        n_controls,
        max_count,
    )
    inputs = _resolve_dgp_inputs(dgp_inputs)
    rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        seed = base_seed + repeat
        for effect_factor in effect_factors:
            pool = _generate_directdgp_pool(
                inputs=inputs,
                n_genes=n_genes,
                n_controls=n_controls,
                cells_per_perturbation=2 * max_count + oracle_pool_size,
                n_perturbations=n_perturbations,
                p_effect=p_effect,
                effect_factor=effect_factor,
                B=B,
                mu_l=mu_l,
                seed=seed,
            )
            perturbation_labels = _ordered_perturbation_labels(pool)
            population_adata = _select_directdgp_pool(
                pool,
                perturbation_labels,
                np.full(n_perturbations, oracle_pool_size, dtype=np.int64),
                offset=2 * max_count,
                include_controls=False,
            )
            population_pseudobulk = _pseudobulk_from_adata(
                population_adata,
                perturbation_labels,
            )

            for sample_size in sample_sizes:
                equal_sizes = np.full(n_perturbations, sample_size, dtype=np.int64)
                adata = _select_directdgp_pool(
                    pool,
                    perturbation_labels,
                    equal_sizes,
                    include_controls=True,
                )
                matched_adata = _select_directdgp_pool(
                    pool,
                    perturbation_labels,
                    equal_sizes,
                    offset=max_count,
                    include_controls=False,
                )
                observed_pseudobulk = _pseudobulk_from_adata(adata, perturbation_labels)
                matched_pseudobulk = _pseudobulk_from_adata(
                    matched_adata,
                    perturbation_labels,
                )
                pca_model, sigma_squared, observed_score = _fit_and_score_observed(
                    adata,
                    observed_pseudobulk,
                    n_splits,
                    null_quantile,
                    target_null_similarity,
                    seed,
                )
                deterministic_score = vendi_score_pseudobulk(
                    population_pseudobulk,
                    pca_model=pca_model,
                    outer_sigma_squared=sigma_squared,
                )
                matched_score = vendi_score_pseudobulk(
                    matched_pseudobulk,
                    pca_model=pca_model,
                    outer_sigma_squared=sigma_squared,
                )
                rows.append(
                    {
                        "experiment": "deterministic_oracle",
                        "sample_size": sample_size,
                        "repeat": repeat,
                        "seed": seed,
                        "n_perturbations": n_perturbations,
                        "n_genes": n_genes,
                        "n_controls": n_controls,
                        "minimum_sample_size": minimum_sample_size,
                        "oracle_pool_size": oracle_pool_size,
                        "n_null": sample_size,
                        "n_splits": n_splits,
                        "null_quantile": null_quantile,
                        "target_null_similarity": target_null_similarity,
                        "p_effect": p_effect,
                        "effect_factor": effect_factor,
                        "B": B,
                        "mu_l": mu_l,
                        "vendi_observed": observed_score,
                        "vendi_deterministic_oracle": deterministic_score,
                        "vendi_sampling_matched_oracle": matched_score,
                        "vendi_ratio_deterministic": deterministic_score / observed_score,
                        "vendi_ratio_sampling_matched": matched_score / observed_score,
                        "sigma_null_squared": sigma_squared,
                    }
                )

    return pd.DataFrame(rows)


def summarize_heterogeneous_null(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize heterogeneous-null results across simulation repeats."""
    return cast(
        pd.DataFrame,
        results.groupby(["effect_factor", "design"], sort=False, as_index=False).agg(
            n_perturbations=("n_perturbations", "first"),
            n_genes=("n_genes", "first"),
            n_controls=("n_controls", "first"),
            minimum_sample_size=("minimum_sample_size", "first"),
            n_min=("n_min", "first"),
            n_median=("n_median", "first"),
            n_max=("n_max", "first"),
            n_cv=("n_cv", "first"),
            n_null=("n_null", "first"),
            n_splits=("n_splits", "first"),
            null_quantile=("null_quantile", "first"),
            target_null_similarity=("target_null_similarity", "first"),
            p_effect=("p_effect", "first"),
            B=("B", "first"),
            mu_l=("mu_l", "first"),
            vendi_observed_mean=("vendi_observed", "mean"),
            vendi_observed_std=("vendi_observed", "std"),
            mean_similarity_mean=("mean_off_diagonal_similarity", "mean"),
            mean_similarity_std=("mean_off_diagonal_similarity", "std"),
            sigma_null_squared_mean=("sigma_null_squared", "mean"),
            n_repeats=("repeat", "count"),
        ),
    )


def summarize_deterministic_oracle(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize deterministic-oracle results across simulation repeats."""
    return cast(
        pd.DataFrame,
        results.groupby(["effect_factor", "sample_size"], sort=True, as_index=False).agg(
            n_perturbations=("n_perturbations", "first"),
            n_genes=("n_genes", "first"),
            n_controls=("n_controls", "first"),
            minimum_sample_size=("minimum_sample_size", "first"),
            oracle_pool_size=("oracle_pool_size", "first"),
            n_null=("n_null", "first"),
            n_splits=("n_splits", "first"),
            null_quantile=("null_quantile", "first"),
            target_null_similarity=("target_null_similarity", "first"),
            p_effect=("p_effect", "first"),
            B=("B", "first"),
            mu_l=("mu_l", "first"),
            vendi_observed_mean=("vendi_observed", "mean"),
            vendi_observed_std=("vendi_observed", "std"),
            vendi_deterministic_oracle_mean=("vendi_deterministic_oracle", "mean"),
            vendi_deterministic_oracle_std=("vendi_deterministic_oracle", "std"),
            vendi_sampling_matched_oracle_mean=("vendi_sampling_matched_oracle", "mean"),
            vendi_sampling_matched_oracle_std=("vendi_sampling_matched_oracle", "std"),
            vendi_ratio_deterministic_mean=("vendi_ratio_deterministic", "mean"),
            vendi_ratio_deterministic_std=("vendi_ratio_deterministic", "std"),
            vendi_ratio_sampling_matched_mean=("vendi_ratio_sampling_matched", "mean"),
            vendi_ratio_sampling_matched_std=("vendi_ratio_sampling_matched", "std"),
            sigma_null_squared_mean=("sigma_null_squared", "mean"),
            n_repeats=("repeat", "count"),
        ),
    )


def plot_sample_size_bias(
    null_summary: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot sample-size sensitivity across DirectDGP effect factors."""
    apply_paper_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    observed_factors = set(null_summary["effect_factor"].to_numpy(dtype=float))
    effect_factors = [factor for factor in _DEFAULT_EFFECT_FACTORS if factor in observed_factors]
    effect_factors.extend(sorted(observed_factors.difference(effect_factors)))
    palette = ("#495057", "#1971c2", "#2b8a3e", "#d94801")
    factor_colors = {
        factor: palette[factor_idx % len(palette)]
        for factor_idx, factor in enumerate(effect_factors)
    }
    positions = np.arange(len(_DESIGN_ORDER))
    ordered_null: pd.DataFrame | None = None
    for effect_factor in effect_factors:
        factor_data = null_summary.loc[null_summary["effect_factor"] == effect_factor]
        ordered_null = cast(
            pd.DataFrame,
            factor_data.set_index("design").loc[list(_DESIGN_ORDER)].reset_index(),
        )
        axes[0].errorbar(
            positions,
            ordered_null["vendi_observed_mean"].to_numpy(dtype=float),
            yerr=np.nan_to_num(
                ordered_null["vendi_observed_std"].to_numpy(dtype=float),
                nan=0.0,
            ),
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=factor_colors[effect_factor],
            label=rf"$\epsilon={effect_factor:g}$",
        )
    if ordered_null is None:
        raise ValueError("null_summary must contain at least one effect factor.")
    axes[0].axhline(1.0, linestyle="--", color="0.4", linewidth=1.3)
    design_names = ordered_null["design"].astype(str).to_numpy()
    minimum_sizes = ordered_null["n_min"].to_numpy(dtype=int)
    maximum_sizes = ordered_null["n_max"].to_numpy(dtype=int)
    imbalance_labels = [
        f"{design_name.title()}\n({minimum_size}-{maximum_size})"
        for design_name, minimum_size, maximum_size in zip(
            design_names,
            minimum_sizes,
            maximum_sizes,
            strict=True,
        )
    ]
    axes[0].set_xticks(positions, imbalance_labels)
    axes[0].set_xlabel(r"Perturbation sample size ($N_q$)")
    axes[0].set_ylabel("Pseudobulk Vendi")
    axes[0].set_axisbelow(True)
    axes[0].grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].legend(title="Effect factor", fontsize=10)

    sample_sizes = np.sort(oracle_summary["sample_size"].unique().astype(float))
    ratio_specs = (
        (
            "vendi_ratio_deterministic_mean",
            "vendi_ratio_deterministic_std",
            "Large-pool mean oracle",
            "#c92a2a",
        ),
        (
            "vendi_ratio_sampling_matched_mean",
            "vendi_ratio_sampling_matched_std",
            "Sampling-matched oracle",
            "#087f5b",
        ),
    )
    oracle_styles = (("-", "o"), ("--", "s"))
    for factor_idx, effect_factor in enumerate(effect_factors):
        factor_data = oracle_summary.loc[
            oracle_summary["effect_factor"] == effect_factor
        ].sort_values("sample_size")
        for oracle_idx, (mean_column, std_column, label, _) in enumerate(ratio_specs):
            linestyle, marker = oracle_styles[oracle_idx]
            axes[1].errorbar(
                factor_data["sample_size"].to_numpy(dtype=float),
                factor_data[mean_column].to_numpy(dtype=float),
                yerr=np.nan_to_num(
                    factor_data[std_column].to_numpy(dtype=float),
                    nan=0.0,
                ),
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                capsize=4,
                color=factor_colors[effect_factor],
                label=label if factor_idx == 0 else "_nolegend_",
            )
    axes[1].axhline(1.0, linestyle="--", color="0.4", linewidth=1.3)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(sample_sizes, [str(int(value)) for value in sample_sizes])
    axes[1].set_xlabel(r"Cells per perturbation ($N_q$)")
    axes[1].set_ylabel("Vendi Ratio")
    axes[1].set_axisbelow(True)
    axes[1].grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axes[1].legend(
        loc="center",
        bbox_to_anchor=(0.71, 0.24),
        fontsize=11,
    )

    for suffix in ("pdf", "png"):
        output_path = output_dir / f"vendi_sample_size_bias.{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved sample-size bias plot to {output_path}")
    plt.close(fig)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    """Parse a comma-separated sequence of integers."""
    return tuple(int(value) for value in text.split(",") if value.strip())


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    """Parse a comma-separated sequence of floats."""
    return tuple(float(value) for value in text.split(",") if value.strip())


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Stress-test Vendi calibration and oracle ratios with DirectDGP."
    )
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    parser.add_argument("--n-perturbations", type=int, default=20)
    parser.add_argument("--n-genes", type=int, default=_DEFAULT_N_GENES)
    parser.add_argument("--n-controls", type=int, default=_DEFAULT_N_CONTROLS)
    parser.add_argument("--reference-size", type=int, default=128)
    parser.add_argument(
        "--minimum-sample-size",
        type=int,
        default=_DEFAULT_MINIMUM_SAMPLE_SIZE,
    )
    parser.add_argument("--sample-sizes", type=_parse_int_tuple, default=_DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--oracle-pool-size", type=int, default=_DEFAULT_ORACLE_POOL_SIZE)
    parser.add_argument("--n-splits", type=int, default=200)
    parser.add_argument("--null-quantile", type=float, default=0.95)
    parser.add_argument("--target-null-similarity", type=float, default=0.95)
    parser.add_argument("--p-effect", type=float, default=_DEFAULT_P_EFFECT)
    parser.add_argument(
        "--effect-factors",
        type=_parse_float_tuple,
        default=_DEFAULT_EFFECT_FACTORS,
    )
    parser.add_argument("--B", type=float, default=_DEFAULT_B)
    parser.add_argument("--mu-l", type=float, default=_DEFAULT_MU_L)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """Run both sample-size experiments and save tables and figures."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    null_results = run_heterogeneous_null_experiment(
        repeats=args.repeats,
        n_perturbations=args.n_perturbations,
        n_genes=args.n_genes,
        n_controls=args.n_controls,
        reference_size=args.reference_size,
        minimum_sample_size=args.minimum_sample_size,
        n_splits=args.n_splits,
        null_quantile=args.null_quantile,
        target_null_similarity=args.target_null_similarity,
        p_effect=args.p_effect,
        effect_factors=args.effect_factors,
        B=args.B,
        mu_l=args.mu_l,
        base_seed=args.seed,
    )
    oracle_results = run_deterministic_oracle_experiment(
        repeats=args.repeats,
        sample_sizes=args.sample_sizes,
        n_perturbations=args.n_perturbations,
        n_genes=args.n_genes,
        n_controls=args.n_controls,
        minimum_sample_size=args.minimum_sample_size,
        oracle_pool_size=args.oracle_pool_size,
        n_splits=args.n_splits,
        null_quantile=args.null_quantile,
        target_null_similarity=args.target_null_similarity,
        p_effect=args.p_effect,
        effect_factors=args.effect_factors,
        B=args.B,
        mu_l=args.mu_l,
        base_seed=args.seed,
    )
    null_summary = summarize_heterogeneous_null(null_results)
    oracle_summary = summarize_deterministic_oracle(oracle_results)

    outputs = {
        "heterogeneous_null.csv": null_results,
        "heterogeneous_null_summary.csv": null_summary,
        "deterministic_oracle.csv": oracle_results,
        "deterministic_oracle_summary.csv": oracle_summary,
    }
    for filename, frame in outputs.items():
        output_path = output_dir / filename
        frame.to_csv(output_path, index=False)
        print(f"Wrote results to {output_path}")

    plot_sample_size_bias(null_summary, oracle_summary, output_dir)


if __name__ == "__main__":
    main()
