import anndata as ad
import numpy as np
import pandas as pd
import pytest

import perturbations.analyses.simulator_validation.generate_statistics as generate_statistics
from perturbations.analyses.simulator_validation.generate_statistics import (
    compute_validation_summary,
    resolve_context_key,
)


@pytest.mark.parametrize("context_key", ["cell_line", "donor_timepoint"])
def test_resolve_context_key_infers_standard_context_columns(context_key: str) -> None:
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame({context_key: ["a", "b"]}, index=["cell_1", "cell_2"]),
    )

    assert resolve_context_key(adata) == context_key


def test_resolve_context_key_prefers_cell_line() -> None:
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"cell_line": ["a", "b"], "context": ["c", "d"]},
            index=["cell_1", "cell_2"],
        ),
    )

    assert resolve_context_key(adata) == "cell_line"


def test_resolve_context_key_requires_context_metadata() -> None:
    adata = ad.AnnData(X=np.ones((1, 1)))

    with pytest.raises(KeyError, match="Could not infer a context column"):
        resolve_context_key(adata)


def test_validation_summary_requires_controls_in_every_context() -> None:
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"cell_line": ["a", "a"], "perturbation": ["gene_1", "gene_1"]},
            index=["cell_1", "cell_2"],
        ),
    )

    with pytest.raises(ValueError, match="No control cells"):
        compute_validation_summary(
            adata,
            name="test",
            include_gene_pairs=False,
            include_perturbation_effects=False,
        )


def test_validation_summary_forwards_workers_and_uses_control_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "cell_line": ["a", "a", "b", "b"],
                "perturbation": ["control", "gene_1", "control", "gene_1"],
            },
            index=["a_control", "a_gene_1", "b_control", "b_gene_1"],
        ),
    )
    baseline_inputs: list[list[str]] = []
    observed_workers: list[int] = []

    def capture_baseline(data: ad.AnnData, layer: str | None = None) -> pd.DataFrame:
        baseline_inputs.append(data.obs_names.tolist())
        return pd.DataFrame({"value": [1.0]})

    def fake_summary(
        stats: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        return pd.DataFrame({"metric": ["value"], "median": [float(stats.iloc[0, 0])]})

    def fake_effects(data: ad.AnnData, **kwargs: object) -> pd.DataFrame:
        observed_workers.append(int(kwargs["context_workers"]))
        return pd.DataFrame({"cell_line": ["a", "b"]})

    def fake_effect_summary(stats: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"metric": ["effect"], "median": [float(len(stats))]})

    monkeypatch.setattr(generate_statistics, "gene_wise_statistics", capture_baseline)
    monkeypatch.setattr(generate_statistics, "cell_wise_statistics", capture_baseline)
    monkeypatch.setattr(generate_statistics, "summarize_statistics", fake_summary)
    monkeypatch.setattr(generate_statistics, "perturbation_effect_statistics", fake_effects)
    monkeypatch.setattr(
        generate_statistics,
        "summarize_perturbation_statistics",
        fake_effect_summary,
    )

    compute_validation_summary(
        adata,
        name="test",
        include_gene_pairs=False,
        context_workers=4,
    )

    assert baseline_inputs == [["a_control"], ["a_control"], ["b_control"], ["b_control"]]
    assert observed_workers == [4]
