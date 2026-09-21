"""Tests for Vendi sample-size calibration stress experiments."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

from pathlib import Path

import numpy as np
import pytest

from perturbations.analyses.vendi_score.vendi_sample_size_bias import (
    plot_sample_size_bias,
    run_deterministic_oracle_experiment,
    run_heterogeneous_null_experiment,
    sample_size_designs,
    summarize_deterministic_oracle,
    summarize_heterogeneous_null,
)


def _dgp_inputs(n_parameters: int = 64) -> dict[str, np.ndarray]:
    """Return compact positive DirectDGP parameters for focused tests."""
    return {
        "all_theta": np.full(n_parameters, 5.0),
        "control_mu": np.linspace(0.05, 1.0, n_parameters),
        "pert_mu": np.linspace(0.08, 1.3, n_parameters),
        "gene_names": np.asarray([f"g{idx}" for idx in range(n_parameters)]),
    }


def test_sample_size_designs_hold_median_fixed() -> None:
    designs = sample_size_designs(n_perturbations=8, reference_size=128)

    assert list(designs) == ["balanced", "moderate", "severe"]
    assert all(np.median(counts) == 128 for counts in designs.values())
    assert all(counts.sum() == designs["balanced"].sum() for counts in designs.values())
    assert all(counts.min() >= 64 for counts in designs.values())
    assert designs["moderate"].min() == 96
    assert designs["severe"].min() == 64
    assert np.std(designs["balanced"]) == 0.0
    assert np.std(designs["moderate"]) < np.std(designs["severe"])


def test_heterogeneous_null_exposes_fixed_size_calibration() -> None:
    results = run_heterogeneous_null_experiment(
        repeats=8,
        n_perturbations=8,
        n_genes=32,
        n_controls=512,
        reference_size=128,
        n_splits=30,
        p_effect=0.1,
        effect_factors=(1.0, 1.5, 2.0, 5.0),
        base_seed=10,
        dgp_inputs=_dgp_inputs(),
    )

    assert (results["minimum_sample_size"] == 64).all()
    assert (results["n_min"] >= 64).all()
    assert (results["n_null"] == 128).all()
    assert (results["B"] == 0.0).all()
    assert (results["p_effect"] == 0.1).all()
    assert set(results["effect_factor"]) == {1.0, 1.5, 2.0, 5.0}

    summary = summarize_heterogeneous_null(results).set_index(["effect_factor", "design"])
    assert (
        summary.loc[(1.0, "severe"), "vendi_observed_mean"]
        > summary.loc[(1.0, "balanced"), "vendi_observed_mean"]
    )
    assert (
        summary.loc[(1.0, "severe"), "mean_similarity_mean"]
        < summary.loc[(1.0, "balanced"), "mean_similarity_mean"]
    )


def test_oracle_ratios_converge_with_sample_size() -> None:
    results = run_deterministic_oracle_experiment(
        repeats=8,
        sample_sizes=(64, 128, 256),
        n_perturbations=8,
        n_genes=32,
        n_controls=512,
        oracle_pool_size=512,
        n_splits=30,
        p_effect=0.1,
        effect_factors=(1.0, 1.5, 2.0, 5.0),
        base_seed=20,
        dgp_inputs=_dgp_inputs(),
    )

    assert np.allclose(
        results["vendi_ratio_deterministic"],
        results["vendi_deterministic_oracle"] / results["vendi_observed"],
    )
    assert (results["minimum_sample_size"] == 64).all()
    assert (results["sample_size"] >= 64).all()
    assert (results["B"] == 0.0).all()
    assert (results["p_effect"] == 0.1).all()
    assert set(results["effect_factor"]) == {1.0, 1.5, 2.0, 5.0}
    summary = summarize_deterministic_oracle(results).set_index(["effect_factor", "sample_size"])
    assert summary.loc[(5.0, 64), "vendi_ratio_deterministic_mean"] < 1.0
    assert summary.loc[(5.0, 64), "vendi_ratio_sampling_matched_mean"] < 1.0
    assert (
        abs(
            summary.loc[(5.0, 64), "vendi_ratio_deterministic_mean"]
            - summary.loc[(5.0, 64), "vendi_ratio_sampling_matched_mean"]
        )
        < 0.01
    )
    assert (
        summary.loc[(5.0, 64), "vendi_ratio_deterministic_mean"]
        < summary.loc[(5.0, 256), "vendi_ratio_deterministic_mean"]
    )
    assert (
        summary.loc[(5.0, 64), "vendi_ratio_sampling_matched_mean"]
        < summary.loc[(5.0, 256), "vendi_ratio_sampling_matched_mean"]
    )


def test_plot_sample_size_bias_writes_pdf_and_png(tmp_path: Path) -> None:
    inputs = _dgp_inputs(16)
    null_results = run_heterogeneous_null_experiment(
        repeats=2,
        n_perturbations=4,
        n_genes=8,
        n_controls=32,
        reference_size=4,
        minimum_sample_size=2,
        n_splits=5,
        p_effect=0.1,
        effect_factors=(1.0, 5.0),
        dgp_inputs=inputs,
    )
    oracle_results = run_deterministic_oracle_experiment(
        repeats=2,
        sample_sizes=(2, 4),
        n_perturbations=4,
        n_genes=8,
        n_controls=16,
        minimum_sample_size=2,
        oracle_pool_size=8,
        n_splits=5,
        p_effect=0.1,
        effect_factors=(1.0, 5.0),
        dgp_inputs=inputs,
    )

    plot_sample_size_bias(
        summarize_heterogeneous_null(null_results),
        summarize_deterministic_oracle(oracle_results),
        tmp_path,
    )

    for suffix in ("pdf", "png"):
        output_path = tmp_path / f"vendi_sample_size_bias.{suffix}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_sample_size_experiments_validate_dimensions() -> None:
    with pytest.raises(ValueError, match="multiple of four"):
        sample_size_designs(n_perturbations=6, reference_size=128)
    with pytest.raises(ValueError, match="exceed minimum_sample_size"):
        sample_size_designs(n_perturbations=8, reference_size=64)
    with pytest.raises(ValueError, match="minimum_sample_size=64"):
        run_deterministic_oracle_experiment(
            repeats=1,
            sample_sizes=(32, 64),
            n_perturbations=4,
            n_genes=8,
            n_controls=128,
            n_splits=5,
        )
    with pytest.raises(ValueError, match="n_genes must be at least"):
        run_heterogeneous_null_experiment(
            repeats=1,
            n_perturbations=4,
            n_genes=3,
            n_controls=32,
            reference_size=4,
            minimum_sample_size=2,
            n_splits=5,
        )
    with pytest.raises(ValueError, match="twice the largest"):
        run_deterministic_oracle_experiment(
            repeats=1,
            sample_sizes=(16,),
            n_perturbations=4,
            n_genes=8,
            n_controls=16,
            minimum_sample_size=16,
            n_splits=5,
        )
