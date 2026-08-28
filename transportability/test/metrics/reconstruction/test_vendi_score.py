"""Unit tests for the run_vendi Vendi + PDS scoring pipeline helpers."""

from __future__ import annotations

import math

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbations.analyses.vendi_score.run_vendi import (
    _OUTPUT_COLUMNS,
    DatasetSpec,
    _base_result_row,
    _split_half_pds,
)


def _make_adata(
    n_cells_per_group: int,
    n_genes: int,
    perturbations: list[str],
    contexts: list[str] | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Build a minimal AnnData with distinct perturbation profiles."""
    rng = np.random.default_rng(seed)
    n_groups = len(perturbations)
    n_total = n_cells_per_group * n_groups
    X = np.empty((n_total, n_genes), dtype=np.float32)
    labels: list[str] = []
    ctx_labels: list[str] = []
    for i, pert in enumerate(perturbations):
        start = i * n_cells_per_group
        end = start + n_cells_per_group
        X[start:end] = rng.normal(loc=i * 2.0, scale=0.1, size=(n_cells_per_group, n_genes))
        labels.extend([pert] * n_cells_per_group)
        if contexts is not None:
            ctx_labels.extend([contexts[i % len(contexts)]] * n_cells_per_group)

    obs = pd.DataFrame({"perturbation": labels})
    if ctx_labels:
        obs["cell_line"] = ctx_labels
    return ad.AnnData(X=X, obs=obs)


# --- _split_half_pds tests ---


class TestSplitHalfPds:
    def test_returns_nan_with_fewer_than_two_non_control_perturbations(self) -> None:
        adata = _make_adata(10, 5, ["control", "gene1"])
        scores = _split_half_pds(adata, layer_key=None, control_label="control", random_state=0)
        assert math.isnan(scores["pds_l1"])
        assert math.isnan(scores["pds_l2"])
        assert math.isnan(scores["pds_cosine"])

    def test_returns_nan_without_control_cells(self) -> None:
        adata = _make_adata(10, 5, ["gene1", "gene2", "gene3"])
        scores = _split_half_pds(adata, layer_key=None, control_label="control", random_state=0)
        assert math.isnan(scores["pds_l1"])

    def test_returns_finite_scores_with_distinct_perturbations(self) -> None:
        adata = _make_adata(20, 10, ["control", "gene1", "gene2", "gene3"], seed=42)
        scores = _split_half_pds(adata, layer_key=None, control_label="control", random_state=42)
        for key in ("pds_l1", "pds_l2", "pds_cosine"):
            assert np.isfinite(scores[key]), f"{key} should be finite"
            assert 0.0 <= scores[key] <= 1.0, f"{key}={scores[key]} out of [0, 1]"

    def test_symmetric_across_splits(self) -> None:
        adata = _make_adata(40, 8, ["control", "gene1", "gene2", "gene3"], seed=7)
        scores = _split_half_pds(adata, layer_key=None, control_label="control", random_state=7)
        # Well-separated synthetic data yields perfect PDS (1.0).
        assert 0.0 < scores["pds_l1"] <= 1.0

    def test_uses_specified_layer(self) -> None:
        adata = _make_adata(20, 5, ["control", "gene1", "gene2", "gene3"], seed=0)
        adata.layers["custom"] = adata.X * 10.0
        scores_x = _split_half_pds(adata, layer_key=None, control_label="control", random_state=0)
        scores_layer = _split_half_pds(
            adata, layer_key="custom", control_label="control", random_state=0
        )
        # Same relative structure scaled 10x — PDS should be identical.
        np.testing.assert_allclose(scores_x["pds_l1"], scores_layer["pds_l1"], atol=1e-6)

    def test_deterministic_with_same_seed(self) -> None:
        adata = _make_adata(30, 6, ["control", "gene1", "gene2", "gene3"], seed=99)
        a = _split_half_pds(adata, layer_key=None, control_label="control", random_state=5)
        b = _split_half_pds(adata, layer_key=None, control_label="control", random_state=5)
        assert a == b

    def test_skips_perturbations_with_single_cell(self) -> None:
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=np.float32)
        obs = pd.DataFrame(
            {"perturbation": ["control", "control", "gene1", "gene1", "gene2", "gene2", "gene3"]}
        )
        adata = ad.AnnData(X=X, obs=obs)
        # gene3 has only 1 cell — should be skipped, leaving gene1 + gene2.
        scores = _split_half_pds(adata, layer_key=None, control_label="control", random_state=0)
        assert np.isfinite(scores["pds_l1"])

    def test_handles_sparse_matrix(self) -> None:
        """Real datasets and causalDGP output store expression as sparse CSR."""
        dense = _make_adata(20, 10, ["control", "gene1", "gene2", "gene3"], seed=42)
        adata = ad.AnnData(X=sparse.csr_matrix(dense.X), obs=dense.obs.copy())
        scores = _split_half_pds(adata, layer_key=None, control_label="control", random_state=42)
        for key in ("pds_l1", "pds_l2", "pds_cosine"):
            assert np.isfinite(scores[key]), f"{key} should be finite for sparse input"

    def test_sparse_matches_dense(self) -> None:
        dense = _make_adata(20, 10, ["control", "gene1", "gene2", "gene3"], seed=42)
        sparse_adata = ad.AnnData(X=sparse.csr_matrix(dense.X), obs=dense.obs.copy())
        dense_scores = _split_half_pds(
            dense, layer_key=None, control_label="control", random_state=42
        )
        sparse_scores = _split_half_pds(
            sparse_adata, layer_key=None, control_label="control", random_state=42
        )
        for key in ("pds_l1", "pds_l2", "pds_cosine"):
            np.testing.assert_allclose(dense_scores[key], sparse_scores[key], atol=1e-10)

    def test_handles_sparse_anndata_view(self) -> None:
        """Context slicing passes an AnnData view whose .X is a SparseCSRMatrixView."""
        dense = _make_adata(20, 10, ["control", "gene1", "gene2", "gene3"], seed=3)
        obs = dense.obs.copy()
        # Interleave contexts within each perturbation so every context keeps all groups.
        obs["cell_line"] = np.where(np.arange(obs.shape[0]) % 2 == 0, "c0", "c1")
        adata = ad.AnnData(X=sparse.csr_matrix(dense.X), obs=obs)
        view = adata[adata.obs["cell_line"].astype(str) == "c0"]
        scores = _split_half_pds(view, layer_key=None, control_label="control", random_state=3)
        assert np.isfinite(scores["pds_l1"])


# --- _base_result_row tests ---


class TestBaseResultRow:
    def _make_spec(self) -> DatasetSpec:
        return DatasetSpec(
            dataset="norman19",
            dataset_variant=None,
            dataset_label="norman19",
            dataset_path="unused",
        )

    def test_contains_all_output_columns(self) -> None:
        obs = pd.DataFrame({"perturbation": ["control", "gene1"], "cell_line": ["K562", "K562"]})
        row = _base_result_row(
            spec=self._make_spec(),
            obs=obs,
            n_vars=100,
            reported_layer_key="X",
            batch_size=64,
            n_pca_components=50,
            sample_size=500,
            random_state=0,
            execution_time_seconds=1.5,
            score=3.14,
        )
        for col in _OUTPUT_COLUMNS:
            if col in ("pds_l1", "pds_l2", "pds_cosine", "vendi_score_pseudobulk"):
                continue  # added by callers, not _base_result_row
            assert col in row, f"Missing column: {col}"

    def test_scope_defaults_to_all(self) -> None:
        obs = pd.DataFrame({"perturbation": ["control", "gene1"], "cell_line": ["K562", "K562"]})
        row = _base_result_row(
            spec=self._make_spec(),
            obs=obs,
            n_vars=10,
            reported_layer_key="X",
            batch_size=32,
            n_pca_components=50,
            sample_size=100,
            random_state=0,
            execution_time_seconds=0.0,
            score=1.0,
        )
        assert row["scope"] == "all"

    def test_non_control_perturbation_count(self) -> None:
        obs = pd.DataFrame(
            {
                "perturbation": ["control", "control", "gene1", "gene2"],
                "cell_line": ["K562"] * 4,
            }
        )
        row = _base_result_row(
            spec=self._make_spec(),
            obs=obs,
            n_vars=5,
            reported_layer_key="X",
            batch_size=32,
            n_pca_components=50,
            sample_size=100,
            random_state=0,
            execution_time_seconds=0.0,
            score=2.0,
        )
        assert row["n_total_perturbations"] == 2


# --- Output columns contract ---


def test_output_columns_include_pds_and_scope() -> None:
    assert "scope" in _OUTPUT_COLUMNS
    assert "pds_l1" in _OUTPUT_COLUMNS
    assert "pds_l2" in _OUTPUT_COLUMNS
    assert "pds_cosine" in _OUTPUT_COLUMNS
    assert "vendi_score_cell" in _OUTPUT_COLUMNS
    assert "vendi_score_pseudobulk" in _OUTPUT_COLUMNS


# --- Synthetic CausalDGP layer selection ---


def _fake_causal_dgp_adata(seed: int = 0) -> ad.AnnData:
    """Mimic causalDGP output: raw counts in .X, log-normalized in the layer."""
    rng = np.random.default_rng(seed)
    perts = ["control"] + [f"g{i}" for i in range(6)]
    n_per, n_genes = 40, 12
    counts = np.empty((n_per * len(perts), n_genes), dtype=np.float32)
    labels: list[str] = []
    for i, pert in enumerate(perts):
        counts[i * n_per : (i + 1) * n_per] = rng.poisson(
            lam=1.0 + 3.0 * i, size=(n_per, n_genes)
        ).astype(np.float32)
        labels.extend([pert] * n_per)
    obs = pd.DataFrame(
        {
            "perturbation": labels,
            "cell_line": np.tile(["0", "1"], len(labels) // 2).astype(str),
        }
    )
    adata = ad.AnnData(X=counts, obs=obs)
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    adata.layers["normalized_log1p"] = np.log1p(counts / totals * 1e4).astype(np.float32)
    return adata


@pytest.fixture
def patched_synthetic(monkeypatch: pytest.MonkeyPatch):
    """Patch CausalDGP generation so layer selection can be tested cheaply."""
    from perturbations.analyses.vendi_score import run_vendi as rv

    monkeypatch.setattr(
        rv, "load_parameter_estimation_inputs", lambda: {"all_theta": None, "gene_names": None}
    )
    monkeypatch.setattr(rv, "causalDGP", lambda **kwargs: (_fake_causal_dgp_adata(), []))
    return rv


def _score_kwargs(**overrides):
    base = dict(
        diversity_type="both",
        n_genes=12,
        n_control=40,
        n_per_perturbation=40,
        n_perturbations=6,
        batch_size=32,
        n_pca_components=3,
        sample_size=64,
        random_state=0,
        by_context=False,
    )
    base.update(overrides)
    return base


class TestSyntheticObsLayer:
    def test_defaults_to_normalized_layer_not_raw_x(self, patched_synthetic) -> None:
        rows = patched_synthetic._compute_synthetic_vendi_score(**_score_kwargs())
        assert all(row["layer_key"] == "normalized_log1p" for row in rows)

    def test_missing_normalized_layer_raises(self, patched_synthetic) -> None:
        def _causal_dgp_no_norm(**kwargs):
            adata = _fake_causal_dgp_adata()
            del adata.layers["normalized_log1p"]
            return adata, []

        patched_synthetic.causalDGP = _causal_dgp_no_norm
        with pytest.raises(KeyError, match="normalized_log1p"):
            patched_synthetic._compute_synthetic_vendi_score(**_score_kwargs())


# --- Real-data layer selection ---


def _make_real_adata() -> ad.AnnData:
    """Build a minimal AnnData mimicking a real dataset with counts + normalized layer."""
    rng = np.random.default_rng(0)
    n_per, n_genes = 20, 8
    perts = ["control", "gene1", "gene2", "gene3"]
    counts = np.empty((n_per * len(perts), n_genes), dtype=np.float32)
    labels: list[str] = []
    for i, p in enumerate(perts):
        counts[i * n_per : (i + 1) * n_per] = rng.poisson(
            lam=2.0 + 3.0 * i, size=(n_per, n_genes)
        ).astype(np.float32)
        labels.extend([p] * n_per)
    obs = pd.DataFrame({"perturbation": labels, "cell_line": ["K562"] * len(labels)})
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    adata.layers["normalized_log1p"] = np.log1p(counts / totals * 1e4).astype(np.float32)
    return adata


class TestRealDataLayerSelection:
    def test_h5ad_defaults_to_normalized_log1p(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from perturbations.analyses.vendi_score import run_vendi as rv

        adata = _make_real_adata()
        monkeypatch.setattr(rv, "load_real_dataset", lambda dataset_path: (adata, {}))
        monkeypatch.setattr(
            rv,
            "validate_perturbation_targets_subset_from_obs",
            lambda obs, gene_names, control_label: None,
        )
        monkeypatch.setattr(rv, "_require_existing_path", lambda path: None)

        spec = DatasetSpec(
            dataset="norman19",
            dataset_variant=None,
            dataset_label="norman19",
            dataset_path="fake.h5ad",
        )
        rows = rv._compute_h5ad_vendi_score(
            spec=spec,
            counts_layer="counts",
            batch_size=32,
            n_pca_components=3,
            sample_size=32,
            random_state=0,
            norm_target_sum=1e4,
            by_context=False,
        )
        assert len(rows) == 1
        assert rows[0]["layer_key"] == "normalized_log1p"
        assert np.isfinite(rows[0]["vendi_score_cell"])

    def test_h5ad_builds_normalized_layer_from_counts_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from perturbations.analyses.vendi_score import run_vendi as rv

        adata = _make_real_adata()
        del adata.layers["normalized_log1p"]

        monkeypatch.setattr(rv, "load_real_dataset", lambda dataset_path: (adata, {}))
        monkeypatch.setattr(
            rv,
            "validate_perturbation_targets_subset_from_obs",
            lambda obs, gene_names, control_label: None,
        )
        monkeypatch.setattr(rv, "_require_existing_path", lambda path: None)

        spec = DatasetSpec(
            dataset="norman19",
            dataset_variant=None,
            dataset_label="norman19",
            dataset_path="fake.h5ad",
        )
        rows = rv._compute_h5ad_vendi_score(
            spec=spec,
            counts_layer="counts",
            batch_size=32,
            n_pca_components=3,
            sample_size=32,
            random_state=0,
            norm_target_sum=1e4,
            by_context=False,
        )
        assert "normalized_log1p" in adata.layers
        assert rows[0]["layer_key"] == "normalized_log1p"
