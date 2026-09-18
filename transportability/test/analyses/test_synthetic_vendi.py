"""Regression coverage for the observed Vendi cache passed by synthetic sweeps."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from perturbations.analyses.synthetic_simulations import random_sweep as sweep
from perturbations.metrics.reconstruction.vendi_score import vendi_score_pseudobulk


@pytest.mark.parametrize("dataset_name,split_strategy", [
    ("directDGP", "in-context"),
    ("causalDGP", "in-context"),
    ("causalDGP", "cross-context"),
])
def test_observed_cache_uses_pseudobulk_reference(
    monkeypatch: pytest.MonkeyPatch, dataset_name: str, split_strategy: str,
) -> None:
    """Exercise the real runner's context preparation and cache forwarding."""
    rng = np.random.default_rng(7)
    labels = np.tile(np.repeat(["control", "g0", "g1", "g2", "g3"], 20), 2)
    contexts = np.repeat([0, 1], 100)
    matrix = rng.poisson(3, size=(len(labels), 4)).astype(np.float32)
    matrix[labels == "g0", 0] += 8
    matrix[labels == "g1", 1] += 5
    matrix[labels == "g2", 2] += 3
    matrix[labels == "g3", 3] += 6
    data = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"perturbation": labels, "cell_line": contexts},
                         index=[f"cell{i}" for i in range(len(labels))]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    data.layers[sweep.NORM_LAYER_KEY] = np.log1p(matrix)
    if dataset_name == "directDGP":
        data = data[data.obs["cell_line"] == 0, :].copy()
    monkeypatch.setattr(sweep, dataset_name, lambda **kwargs: (data, None))
    monkeypatch.setattr(sweep, "MODELS", ("Control", "Average"))
    monkeypatch.setattr(sweep, "get_data_stats", lambda **kwargs: {})
    monkeypatch.setattr(sweep, "scanpy_de_table", lambda **kwargs: None)
    monkeypatch.setattr(sweep, "de_table_to_deg_masks", lambda **kwargs: [])
    monkeypatch.setattr(sweep, "true_degs_for_context", lambda **kwargs: None)
    monkeypatch.setattr(sweep, "build_perturbation_id_map", lambda obs: {})
    monkeypatch.setattr(sweep, "fit_control_incremental_pca", lambda **kwargs: object())
    monkeypatch.setattr(sweep, "estimate_mmd_gamma", lambda **kwargs: 2.0)
    monkeypatch.setattr(sweep, "estimate_vendi_outer_sigma_squared", lambda **kwargs: 3.0)
    legacy_calls = []

    def legacy_score(**kwargs):
        legacy_calls.append(kwargs)
        return 0.25

    monkeypatch.setattr(sweep, "vendi_score", legacy_score)
    evaluated = []

    def check_evaluation(**kwargs):
        expected = vendi_score_pseudobulk(
            kwargs["mu_obs"],
            pca_model=kwargs["vendi_pseudobulk_pca_model"],
            outer_sigma_squared=kwargs["vendi_pseudobulk_sigma_squared"],
        )
        assert kwargs["vendi_score_obs"] == pytest.approx(expected, rel=1e-12)
        assert expected > 1.0
        evaluated.append((kwargs["model"], kwargs["vendi_score_obs"]))
        return {"vendi_score_obs": kwargs["vendi_score_obs"]}

    monkeypatch.setattr(sweep, "evaluation", check_evaluation)
    result = sweep.simulate_one_run(
        dataset_name=dataset_name, split_strategy=split_strategy, diversity_type="both",
        G=4, N0=20, Nk=20, P=4, p_effect=0.1, effect_factor=2.0, B=0.5, mu_l=1.0,
        all_theta=np.ones(4), control_mu=np.ones(4), pert_mu=np.ones(4),
        gene_names=np.asarray(data.var_names), pid=1, trial_id_for_rng=0,
    )
    assert result and evaluated
    assert {row["model"] for row in result} == {"Control", "Average"}
    assert not legacy_calls, "Observed model metrics must not call the MMD-based Vendi function"
