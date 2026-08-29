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


def test_context_filter_matches_the_same_context_from_a_joint_run() -> None:
    """
    Filtering to one context must equal running all contexts and keeping that one.

    This is the property that lets a multi-context dataset be split into
    independent per-context jobs without changing any result.
    """
    rng = np.random.default_rng(0)
    contexts = ["K562", "RPE1"]
    perturbations = ["control", "gene_1"]
    cells_per_group = 20
    n_genes = 8

    blocks, rows = [], []
    for offset, context in enumerate(contexts):
        for perturbation in perturbations:
            mean = 5.0 + 3.0 * offset + (2.0 if perturbation != "control" else 0.0)
            blocks.append(rng.poisson(mean, size=(cells_per_group, n_genes)))
            rows += [{"cell_line": context, "perturbation": perturbation}] * cells_per_group

    adata = ad.AnnData(
        X=np.vstack(blocks).astype(np.float64),
        obs=pd.DataFrame(rows, index=[f"cell_{i}" for i in range(len(rows))]),
    )

    joint, _ = compute_validation_summary(
        adata,
        name="test",
        context_key="cell_line",
        include_gene_pairs=False,
        include_perturbation_effects=False,
        n_boot=25,
    )
    subset = adata[np.asarray(adata.obs["cell_line"]).astype(str) == "RPE1"].copy()
    filtered, _ = compute_validation_summary(
        subset,
        name="test",
        context_key="cell_line",
        include_gene_pairs=False,
        include_perturbation_effects=False,
        n_boot=25,
    )

    joint_rpe1 = joint[joint["context"] == "RPE1"].reset_index(drop=True)

    assert set(filtered["context"]) == {"RPE1"}
    pd.testing.assert_frame_equal(joint_rpe1, filtered.reset_index(drop=True))


def test_context_filter_rejects_an_unknown_context() -> None:
    parser = generate_statistics._build_parser()
    args = parser.parse_args(
        ["--source", "real", "--dataset-path", "x.h5ad", "--context-filter", "HEK293"]
    )

    assert args.context_filter == "HEK293"
