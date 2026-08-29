"""Unit tests for perturbations.analyses.evaluator."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from perturbations.analyses import evaluator


def _make_adata(perturbations: list[str]) -> ad.AnnData:
    """Build a minimal AnnData with a 'perturbation' obs column."""
    n_cells = len(perturbations)
    X = np.arange(n_cells * 2, dtype=np.float32).reshape(n_cells, 2)
    return ad.AnnData(X=X, obs=pd.DataFrame({"perturbation": perturbations}))


@pytest.fixture(autouse=True)
def _stub_unrelated_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every metric unrelated to Vendi scoring so tests stay hermetic."""
    monkeypatch.setattr(evaluator, "distribution_distance", lambda **kwargs: 0.0)
    monkeypatch.setattr(evaluator, "pearson_pert", lambda *a, **k: 0.0)
    monkeypatch.setattr(evaluator, "mean_error_pert", lambda *a, **k: 0.0)
    monkeypatch.setattr(evaluator, "r2_score_pert", lambda *a, **k: 0.0)
    monkeypatch.setattr(evaluator, "pds", lambda *a, **k: 0.0)


def _run_evaluation(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> tuple[dict, list]:
    """Call evaluation() with a recording vendi_score_pseudobulk stub."""
    calls: list[tuple[int, int | None]] = []

    def recording_pseudobulk(pseudobulk, control_idx=None, *, pca_model, outer_sigma_squared):
        calls.append((np.asarray(pseudobulk).shape[0], control_idx))
        return float(len(calls))

    monkeypatch.setattr(evaluator, "vendi_score_pseudobulk", recording_pseudobulk)

    kwargs: dict[str, object] = {
        "mu_control_obs": np.zeros((1, 2), dtype=np.float32),
        "mu_pool_obs": None,
        "true_DEGs": None,
        "obs_DE_table": None,
        "obs_DEGs": [np.zeros(2, dtype=bool), np.zeros(2, dtype=bool)],
        "mmd_gamma": 1.0,
        "mmd_pca_model": None,
        "vendi_outer_sigma_squared": 1.0,
        "vendi_pseudobulk_pca_model": object(),
        "vendi_pseudobulk_sigma_squared": 1.0,
        "perturbation_ids": np.array(["gA", "gB"]),
        "layer_name": None,
        "control_label": -1,
    }
    kwargs.update(overrides)
    result = evaluator.evaluation(**kwargs)
    return result, calls


class TestEvaluationPseudobulkVendi:
    """Verify pseudobulk Vendi is computed for every model type, not just pseudobulk-only ones."""

    def test_cell_level_models_still_use_pseudobulk_vendi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pred = _make_adata(["gA", "gA", "gB", "gB"])
        obs = _make_adata(["gA", "gA", "gB", "gB"])
        mu_pred = np.ones((2, 2), dtype=np.float32)
        mu_obs = np.ones((2, 2), dtype=np.float32) * 2.0

        result, calls = _run_evaluation(
            monkeypatch, pred=pred, obs=obs, mu_pred=mu_pred, mu_obs=mu_obs
        )

        # Vendi is scored from mu_pred/mu_obs (2 calls), never from cell-level AnnData.
        assert calls == [(2, None), (2, None)]
        assert result["vendi_score_pred"] == 1.0
        assert result["vendi_score_obs"] == 2.0

    def test_cell_level_models_honor_control_row_in_mu_pred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pred = _make_adata(["gA", "gA", "gB", "gB"])
        obs = _make_adata(["gA", "gA", "gB", "gB"])
        mu_pred = np.ones((3, 2), dtype=np.float32)  # extra row = control
        mu_obs = np.ones((2, 2), dtype=np.float32)

        _, calls = _run_evaluation(monkeypatch, pred=pred, obs=obs, mu_pred=mu_pred, mu_obs=mu_obs)

        assert calls[0] == (3, 0)

    def test_pseudobulk_only_models_use_pseudobulk_vendi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obs = _make_adata(["gA", "gA", "gB", "gB"])
        mu_pred = np.ones((2, 2), dtype=np.float32)
        mu_obs = np.ones((2, 2), dtype=np.float32) * 2.0

        result, calls = _run_evaluation(
            monkeypatch, pred=None, obs=obs, mu_pred=mu_pred, mu_obs=mu_obs
        )

        assert calls == [(2, None), (2, None)]
        assert np.isnan(result["parametric_distance"])
        assert np.isnan(result["mmd_distance"])
        assert np.isnan(result["fid_distance"])

    def test_vendi_score_obs_override_skips_recomputation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pred = _make_adata(["gA", "gA", "gB", "gB"])
        obs = _make_adata(["gA", "gA", "gB", "gB"])
        mu_pred = np.ones((2, 2), dtype=np.float32)
        mu_obs = np.ones((2, 2), dtype=np.float32)

        result, calls = _run_evaluation(
            monkeypatch,
            pred=pred,
            obs=obs,
            mu_pred=mu_pred,
            mu_obs=mu_obs,
            vendi_score_obs=0.5,
        )

        # Only mu_pred is scored; the pre-supplied vendi_score_obs is reused as-is.
        assert calls == [(2, None)]
        assert result["vendi_score_obs"] == 0.5
