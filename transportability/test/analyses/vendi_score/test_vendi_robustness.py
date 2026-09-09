import numpy as np
import pandas as pd
import pytest

from perturbations.analyses.vendi_score.vendi_robustness import (
    _compute_effective_rank_pseudobulk,
    summarize_sensitivity,
)


def test_effective_rank_pseudobulk_excludes_control() -> None:
    perturbations = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    with_control = np.insert(perturbations, 2, [100.0, 100.0], axis=0)

    score = _compute_effective_rank_pseudobulk(with_control, control_idx=2)

    assert score == pytest.approx(2.0)


def test_summarize_sensitivity_includes_effective_rank() -> None:
    results = pd.DataFrame(
        {
            "dataset": ["test", "test"],
            "sweep": ["noise", "noise"],
            "value": [1.0, 1.0],
            "noise_type": ["gaussian", "gaussian"],
            "noise_variance_fraction": [0.5, 0.5],
            "seed": [0, 1],
            "vendi_cell": [2.0, 4.0],
            "vendi_pseudobulk": [3.0, 5.0],
            "effective_rank_pseudobulk": [4.0, 6.0],
            "pds_l1": [0.2, 0.4],
        }
    )

    summary = summarize_sensitivity(results).iloc[0]

    assert summary["effective_rank_pseudobulk_mean"] == pytest.approx(5.0)
    assert summary["effective_rank_pseudobulk_std"] == pytest.approx(np.sqrt(2.0))
    assert summary["effective_rank_pseudobulk_cv"] == pytest.approx(np.sqrt(2.0) / 5.0)
