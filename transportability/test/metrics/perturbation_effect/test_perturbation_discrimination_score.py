import numpy as np
import pytest

from perturbations.metrics.perturbation_effect.perturbation_discrimination_score import pds


def test_log_fold_matches_manual_log_transformation() -> None:
    observed = np.array([[20.0, 10.0], [10.0, 20.0], [5.0, 10.0]])
    predicted = np.array([[18.0, 11.0], [11.0, 18.0], [6.0, 9.0]])
    reference = np.array([10.0, 10.0])
    eps = 1e-6

    score = pds(observed, predicted, reference, metric="l1", log_fold=True, eps=eps)
    expected = pds(
        np.log2(observed + eps),
        np.log2(predicted + eps),
        np.log2(reference + eps),
        metric="l1",
    )

    assert score == pytest.approx(expected)
