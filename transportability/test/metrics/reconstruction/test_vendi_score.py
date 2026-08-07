from itertools import pairwise

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata.experimental import AnnCollection
from sklearn.decomposition import PCA, IncrementalPCA

from perturbations.metrics.reconstruction.distance_util import adaptive_gamma
from perturbations.metrics.reconstruction.vendi_score import (
    estimate_vendi_outer_sigma_squared,
    estimate_vendi_pseudobulk_sigma_squared,
    fit_vendi_pseudobulk_pca,
    vendi_score,
    vendi_score_pseudobulk,
)


def _adata(labels: list[int], n_features: int = 16, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(labels), n_features)).astype(np.float32)
    obs = pd.DataFrame(
        {"perturbation": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_features)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _calibrated_cell_data(
    separation: float = 1.0,
    n_perturbations: int = 5,
    n_controls: int = 400,
    n_cells: int = 80,
    n_features: int = 10,
    seed: int = 0,
) -> tuple[ad.AnnData, IncrementalPCA, float, float]:
    rng = np.random.default_rng(seed)
    controls = rng.normal(size=(n_controls, n_features))
    pca_model = IncrementalPCA(n_components=n_features).fit(controls)
    directions = pca_model.components_[:n_perturbations]
    groups = [
        rng.normal(size=(n_cells, n_features)) + separation * directions[group_idx]
        for group_idx in range(n_perturbations)
    ]

    matrix = np.vstack([controls, *groups]).astype(np.float32)
    labels = np.concatenate(
        [
            np.repeat("control", n_controls),
            np.concatenate(
                [np.repeat(f"pert_{group_idx}", n_cells) for group_idx in range(n_perturbations)]
            ),
        ]
    )
    obs = pd.DataFrame(
        {"perturbation": labels},
        index=[f"cell_{idx}" for idx in range(matrix.shape[0])],
    )
    var = pd.DataFrame(index=[f"g{idx}" for idx in range(n_features)])
    data = ad.AnnData(X=matrix, obs=obs, var=var)
    layer_key = "normalized"
    data.layers[layer_key] = matrix.copy()

    perturbation_cells = np.vstack(groups)
    gamma = adaptive_gamma(pca_model.transform(perturbation_cells))
    outer_sigma_squared = estimate_vendi_outer_sigma_squared(
        ac=data,
        gamma=gamma,
        pca_model=pca_model,
        layer_key=layer_key,
        control_label="control",
        n_splits=100,
        random_state=seed,
    )
    return data, pca_model, gamma, outer_sigma_squared


def _calibrated_pseudobulk_data(
    n_perturbations: int = 5,
    n_controls: int = 400,
    n_cells: int = 80,
    n_features: int = 10,
    seed: int = 0,
) -> tuple[ad.AnnData, np.ndarray, PCA, float]:
    rng = np.random.default_rng(seed)
    controls = rng.normal(size=(n_controls, n_features))
    groups = [rng.normal(size=(n_cells, n_features)) for _ in range(n_perturbations)]
    pseudobulk = np.vstack([group.mean(axis=0) for group in groups])

    matrix = np.vstack([controls, *groups]).astype(np.float32)
    labels = np.concatenate(
        [
            np.repeat("control", n_controls),
            np.concatenate(
                [np.repeat(f"pert_{group_idx}", n_cells) for group_idx in range(n_perturbations)]
            ),
        ]
    )
    data = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"perturbation": labels},
            index=[f"cell_{idx}" for idx in range(matrix.shape[0])],
        ),
        var=pd.DataFrame(index=[f"g{idx}" for idx in range(n_features)]),
    )
    data.layers["normalized"] = matrix.copy()

    pca_model = fit_vendi_pseudobulk_pca(
        pseudobulk,
        n_pca_components=n_perturbations,
        random_state=seed,
    )
    outer_sigma_squared = estimate_vendi_pseudobulk_sigma_squared(
        ac=data,
        pca_model=pca_model,
        layer_key="normalized",
        control_label="control",
        n_splits=100,
        random_state=seed,
    )
    return data, pseudobulk, pca_model, outer_sigma_squared


# ---------------------------------------------------------------------------
# vendi_score_pseudobulk
# ---------------------------------------------------------------------------


def test_pseudobulk_identical_rows_returns_one() -> None:
    _, _, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(
        n_features=8,
        seed=1,
    )
    pseudobulk = np.ones((5, 8), dtype=np.float64)
    reference = np.zeros((5, 8), dtype=np.float64)
    reference[:, :5] = np.eye(5)
    pca_model = fit_vendi_pseudobulk_pca(reference, n_pca_components=5)
    assert (
        vendi_score_pseudobulk(
            pseudobulk,
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )
        == 1.0
    )


def test_pseudobulk_single_perturbation_returns_one() -> None:
    pseudobulk = np.arange(8, dtype=np.float64).reshape(1, 8)
    assert (
        vendi_score_pseudobulk(
            pseudobulk,
            pca_model=None,
            outer_sigma_squared=None,
        )
        == 1.0
    )


def test_pseudobulk_distinct_rows_bounded_by_count() -> None:
    _, pseudobulk, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(
        n_perturbations=6,
        n_features=20,
        seed=3,
    )
    score = vendi_score_pseudobulk(
        pseudobulk,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    assert 1.0 <= score <= 6.0 + 1e-9


def test_pseudobulk_multiple_distinct_rows_require_pca() -> None:
    _, pseudobulk, _, outer_sigma_squared = _calibrated_pseudobulk_data(seed=11)
    with pytest.raises(ValueError, match="pca_model is required"):
        vendi_score_pseudobulk(
            pseudobulk,
            pca_model=None,
            outer_sigma_squared=outer_sigma_squared,
        )


def test_pseudobulk_multiple_distinct_rows_require_sigma() -> None:
    _, pseudobulk, pca_model, _ = _calibrated_pseudobulk_data(seed=12)
    with pytest.raises(ValueError, match="outer_sigma_squared must be finite"):
        vendi_score_pseudobulk(
            pseudobulk,
            pca_model=pca_model,
            outer_sigma_squared=None,
        )


def test_pseudobulk_zero_separation_and_monotonicity() -> None:
    n_perturbations = 5
    _, base_pseudobulk, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(
        n_perturbations=n_perturbations,
        seed=13,
    )
    scores = [
        vendi_score_pseudobulk(
            base_pseudobulk + separation * pca_model.components_[:n_perturbations],
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )
        for separation in (0.0, 1.0, 2.0, 5.0, 500.0)
    ]

    assert scores[0] == pytest.approx(1.0, abs=0.25)
    assert all(left < right for left, right in pairwise(scores))
    assert scores[-1] == pytest.approx(n_perturbations, abs=0.25)


def test_pseudobulk_control_row_is_excluded() -> None:
    _, perturbations, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(seed=4)
    control = np.full((1, perturbations.shape[1]), 100.0)
    with_control = np.vstack([control, perturbations])

    score_with_control = vendi_score_pseudobulk(
        with_control,
        control_idx=0,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    score_without_control = vendi_score_pseudobulk(
        perturbations,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    assert score_with_control == pytest.approx(score_without_control)


def test_pseudobulk_negative_control_idx() -> None:
    _, perturbations, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(
        n_perturbations=4,
        seed=5,
    )
    control = np.full((1, 10), -50.0)
    pseudobulk = np.vstack([perturbations, control])

    score_neg = vendi_score_pseudobulk(
        pseudobulk,
        control_idx=-1,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    score_pos = vendi_score_pseudobulk(
        pseudobulk,
        control_idx=4,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    assert score_neg == pytest.approx(score_pos)


def test_pseudobulk_reproducible() -> None:
    _, pseudobulk, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(
        n_perturbations=7,
        seed=6,
    )
    kwargs = {
        "pseudobulk": pseudobulk,
        "pca_model": pca_model,
        "outer_sigma_squared": outer_sigma_squared,
    }
    a = vendi_score_pseudobulk(**kwargs)
    b = vendi_score_pseudobulk(**kwargs)
    assert a == b


def test_pseudobulk_rejects_non_2d() -> None:
    _, _, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(seed=7)
    with pytest.raises(ValueError, match="2D array"):
        vendi_score_pseudobulk(
            np.zeros(8),
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )


def test_pseudobulk_rejects_non_finite() -> None:
    _, _, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(seed=8)
    bad = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="non-finite"):
        vendi_score_pseudobulk(
            bad,
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )


def test_pseudobulk_pca_rejects_non_positive_components() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fit_vendi_pseudobulk_pca(np.eye(3, 4), n_pca_components=0)


def test_pseudobulk_control_idx_out_of_bounds() -> None:
    _, pseudobulk, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(seed=9)
    with pytest.raises(IndexError, match="out of bounds"):
        vendi_score_pseudobulk(
            pseudobulk,
            control_idx=20,
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )


def test_pseudobulk_empty_returns_nan() -> None:
    _, _, pca_model, outer_sigma_squared = _calibrated_pseudobulk_data(seed=10)
    assert np.isnan(
        vendi_score_pseudobulk(
            np.zeros((0, 4)),
            pca_model=pca_model,
            outer_sigma_squared=outer_sigma_squared,
        )
    )


# ---------------------------------------------------------------------------
# vendi_score (cell-level)
# ---------------------------------------------------------------------------


def test_cell_level_counts_distinct_perturbations() -> None:
    adata, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(
        separation=5.0,
        seed=5,
    )
    score = vendi_score(
        ac=adata,
        ac_batch_size=64,
        layer_key="normalized",
        control_label="control",
        gamma=gamma,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    assert 1.0 <= score <= 5.0 + 1e-6


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_cell_level_identical_groups_returns_one() -> None:
    rng = np.random.default_rng(6)
    controls = rng.normal(size=(40, 8))
    identical = np.ones((20, 8), dtype=np.float64)
    matrix = np.vstack([controls, identical, identical, identical]).astype(np.float32)
    labels = np.concatenate(
        [np.repeat("control", 40), np.repeat(["pert_0", "pert_1", "pert_2"], 20)]
    )
    obs = pd.DataFrame(
        {"perturbation": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    adata.layers["normalized"] = matrix.copy()
    pca_model = IncrementalPCA(n_components=4).fit(controls)
    score = vendi_score(
        ac=adata,
        ac_batch_size=32,
        layer_key="normalized",
        control_label="control",
        gamma=0.1,
        pca_model=pca_model,
        outer_sigma_squared=1.0,
    )
    assert score == pytest.approx(1.0)


def test_cell_level_score_upper_bounded_by_num_perturbations() -> None:
    adata, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(
        separation=5.0,
        n_perturbations=3,
        seed=12,
    )
    score = vendi_score(
        ac=adata,
        ac_batch_size=64,
        layer_key="normalized",
        control_label="control",
        gamma=gamma,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    assert 1.0 <= score <= 3.0 + 1e-6


def test_cell_level_zero_separation_and_monotonicity() -> None:
    n_perturbations = 5
    reference, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(
        separation=0.0,
        n_perturbations=n_perturbations,
        seed=13,
    )
    labels = np.asarray(reference.obs["perturbation"])
    base_matrix = np.asarray(reference.layers["normalized"], dtype=np.float64)

    scores: list[float] = []
    for separation in (0.0, 1.0, 2.0, 5.0, 500.0):
        data = reference.copy()
        matrix = base_matrix.copy()
        for group_idx in range(n_perturbations):
            group_mask = labels == f"pert_{group_idx}"
            matrix[group_mask] += separation * pca_model.components_[group_idx]
        data.layers["normalized"] = matrix.astype(np.float32)

        scores.append(
            vendi_score(
                ac=data,
                layer_key="normalized",
                control_label="control",
                gamma=gamma,
                pca_model=pca_model,
                outer_sigma_squared=outer_sigma_squared,
            )
        )

    assert scores[0] == pytest.approx(1.0, abs=0.25)
    assert all(left < right for left, right in pairwise(scores))
    assert scores[-1] == pytest.approx(n_perturbations, abs=0.25)


def test_cell_level_excludes_control_label() -> None:
    adata, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(
        separation=2.0,
        n_perturbations=2,
        seed=6,
    )
    common = {
        "ac": adata,
        "ac_batch_size": 64,
        "layer_key": "normalized",
        "gamma": gamma,
        "pca_model": pca_model,
        "outer_sigma_squared": outer_sigma_squared,
    }
    score_excl = vendi_score(control_label="control", **common)
    score_none = vendi_score(control_label=None, **common)
    assert score_none > score_excl


def test_cell_level_reproducible() -> None:
    adata, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(seed=7)
    kwargs = {
        "ac": adata,
        "ac_batch_size": 64,
        "layer_key": "normalized",
        "control_label": "control",
        "gamma": gamma,
        "pca_model": pca_model,
        "outer_sigma_squared": outer_sigma_squared,
    }
    a = vendi_score(**kwargs)
    b = vendi_score(**kwargs)
    assert a == pytest.approx(b)


def test_cell_level_requires_perturbation_column() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(10, 4)).astype(np.float32)
    obs = pd.DataFrame({"other": list(range(10))}, index=[f"c{i}" for i in range(10)])
    adata = ad.AnnData(X=X, obs=obs)
    with pytest.raises(KeyError, match="perturbation"):
        vendi_score(adata, outer_sigma_squared=1.0)


def test_cell_level_no_perturbations_returns_nan() -> None:
    labels = [-1] * 20
    adata = _adata(labels, seed=9)
    assert np.isnan(
        vendi_score(
            adata,
            ac_batch_size=64,
            control_label=-1,
            outer_sigma_squared=None,
        )
    )


def test_cell_level_single_perturbation_returns_one_without_calibration() -> None:
    labels = [-1] * 20 + [0] * 20
    adata = _adata(labels, seed=10)
    assert (
        vendi_score(
            adata,
            control_label=-1,
            gamma=None,
            pca_model=None,
            outer_sigma_squared=None,
        )
        == 1.0
    )


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_cell_level_uses_layer_when_layer_key_given() -> None:
    adata, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(
        separation=2.0,
        n_perturbations=3,
        seed=20,
    )
    adata.layers["flat"] = np.ones_like(adata.layers["normalized"])
    common = {
        "ac": adata,
        "control_label": "control",
        "gamma": gamma,
        "pca_model": pca_model,
        "outer_sigma_squared": outer_sigma_squared,
    }
    from_normalized = vendi_score(layer_key="normalized", **common)
    from_flat = vendi_score(layer_key="flat", **common)
    assert from_normalized > 1.5
    assert from_flat == pytest.approx(1.0)


def test_cell_level_explicit_gamma_skips_estimation() -> None:
    adata, pca_model, _, outer_sigma_squared = _calibrated_cell_data(
        n_perturbations=3,
        seed=22,
    )
    kwargs = {
        "ac": adata,
        "layer_key": "normalized",
        "control_label": "control",
        "gamma": 0.05,
        "pca_model": pca_model,
        "outer_sigma_squared": outer_sigma_squared,
    }
    a = vendi_score(**kwargs)
    b = vendi_score(**kwargs)
    assert a == pytest.approx(b)
    assert 1.0 <= a <= 3.0 + 1e-6


def test_cell_level_accepts_precomputed_pca_model() -> None:
    adata, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(
        n_perturbations=2,
        seed=23,
    )
    score = vendi_score(
        ac=adata,
        layer_key="normalized",
        control_label="control",
        gamma=gamma,
        pca_model=pca_model,
        outer_sigma_squared=outer_sigma_squared,
    )
    assert 1.0 <= score <= 2.0 + 1e-6


def test_cell_level_anncollection_matches_anndata() -> None:
    adata, pca_model, gamma, outer_sigma_squared = _calibrated_cell_data(
        n_perturbations=3,
        seed=24,
    )
    collection = AnnCollection([adata])
    common = {
        "ac_batch_size": 64,
        "layer_key": "normalized",
        "control_label": "control",
        "gamma": gamma,
        "pca_model": pca_model,
        "outer_sigma_squared": outer_sigma_squared,
    }
    from_adata = vendi_score(ac=adata, **common)
    from_collection = vendi_score(ac=collection, **common)
    assert from_collection == pytest.approx(from_adata, rel=1e-6)


def _tight_cluster_dataset(
    n_perturbations: int,
    n_features: int = 20,
    n_cells: int = 60,
    mean_scale: float = 5.0,
    within_scale: float = 1e-3,
    seed: int = 0,
) -> tuple[ad.AnnData, np.ndarray]:
    """
    Build matched cell-level and pseudobulk data for tight clusters.

    Each perturbation (plus a control at row 0 / label -1) is a tight Gaussian
    cluster around a distinct random mean. Returns the AnnData and the
    corresponding pseudobulk matrix (control row first).
    """
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n_perturbations, n_features)) * mean_scale
    labels: list[int] = []
    blocks: list[np.ndarray] = []
    pseudobulk_rows: list[np.ndarray] = []

    control = rng.normal(0.0, within_scale, size=(n_cells, n_features))
    labels += [-1] * n_cells
    blocks.append(control)
    pseudobulk_rows.append(control.mean(axis=0))

    for g in range(n_perturbations):
        block = rng.normal(0.0, within_scale, size=(n_cells, n_features)) + means[g]
        blocks.append(block)
        labels += [g] * n_cells
        pseudobulk_rows.append(block.mean(axis=0))

    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame(
        {"perturbation": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_features)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata, np.vstack(pseudobulk_rows)


def test_cell_level_and_pseudobulk_agree_on_diversity_ordering() -> None:
    # For tight single-cluster perturbations the two metrics use different
    # bandwidth heuristics, so their absolute values differ. They must still
    # agree directionally: a low-diversity dataset (few clusters) scores below a
    # high-diversity dataset (many clusters) under *both* metrics, and both stay
    # within [1, n_perturbations].
    low_adata, low_pca, low_gamma, low_sigma = _calibrated_cell_data(
        separation=5.0,
        n_perturbations=2,
        seed=30,
    )
    high_adata, high_pca, high_gamma, high_sigma = _calibrated_cell_data(
        separation=5.0,
        n_perturbations=6,
        seed=30,
    )

    def pseudobulk(data: ad.AnnData) -> np.ndarray:
        labels = np.asarray(data.obs["perturbation"])
        return np.vstack(
            [
                np.asarray(data.layers["normalized"])[labels == label].mean(axis=0)
                for label in np.unique(labels)
            ]
        )

    low_cell = vendi_score(
        ac=low_adata,
        layer_key="normalized",
        control_label="control",
        gamma=low_gamma,
        pca_model=low_pca,
        outer_sigma_squared=low_sigma,
    )
    high_cell = vendi_score(
        ac=high_adata,
        layer_key="normalized",
        control_label="control",
        gamma=high_gamma,
        pca_model=high_pca,
        outer_sigma_squared=high_sigma,
    )
    low_pb = pseudobulk(low_adata)
    high_pb = pseudobulk(high_adata)
    low_pbulk_pca = fit_vendi_pseudobulk_pca(low_pb, control_idx=0, n_pca_components=10)
    high_pbulk_pca = fit_vendi_pseudobulk_pca(high_pb, control_idx=0, n_pca_components=10)
    low_pbulk_sigma = estimate_vendi_pseudobulk_sigma_squared(
        ac=low_adata,
        pca_model=low_pbulk_pca,
        layer_key="normalized",
    )
    high_pbulk_sigma = estimate_vendi_pseudobulk_sigma_squared(
        ac=high_adata,
        pca_model=high_pbulk_pca,
        layer_key="normalized",
    )
    low_pbulk = vendi_score_pseudobulk(
        low_pb,
        control_idx=0,
        pca_model=low_pbulk_pca,
        outer_sigma_squared=low_pbulk_sigma,
    )
    high_pbulk = vendi_score_pseudobulk(
        high_pb,
        control_idx=0,
        pca_model=high_pbulk_pca,
        outer_sigma_squared=high_pbulk_sigma,
    )

    # Both metrics rank higher diversity above lower diversity.
    assert low_cell < high_cell
    assert low_pbulk < high_pbulk

    # Both stay within their theoretical [1, n] bounds.
    assert 1.0 <= low_cell <= 2.0 + 1e-6
    assert 1.0 <= high_cell <= 6.0 + 1e-6
    assert 1.0 <= low_pbulk <= 2.0 + 1e-6
    assert 1.0 <= high_pbulk <= 6.0 + 1e-6
