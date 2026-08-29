from concurrent.futures import Future

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import perturbations.metrics.perturbation_effect.trade as trade
from perturbations.metrics.perturbation_effect.trade import (
    perturbation_effect_statistics,
    summarize_perturbation_statistics,
    transcriptome_wide_impact,
)


def test_transcriptome_wide_impact_reports_square_root() -> None:
    result = transcriptome_wide_impact(
        lfc=np.array([-1.0, 0.5, 1.5]),
        lfc_se=np.array([0.1, 0.2, 0.1]),
    )

    assert result["sqrt_transcriptome_wide_impact"] == pytest.approx(
        np.sqrt(result["transcriptome_wide_impact"])
    )


def test_summary_reports_square_root_impact_by_default() -> None:
    stats = pd.DataFrame(
        {
            "transcriptome_wide_impact": [1.0, 4.0, 9.0],
            "sqrt_transcriptome_wide_impact": [1.0, 2.0, 3.0],
        }
    )

    summary = summarize_perturbation_statistics(stats, n_boot=20)

    assert summary["metric"].tolist() == [
        "transcriptome_wide_impact",
        "sqrt_transcriptome_wide_impact",
    ]
    sqrt_summary = summary.loc[summary["metric"] == "sqrt_transcriptome_wide_impact"]
    assert sqrt_summary["median"].item() == pytest.approx(2.0)


def test_context_parallelism_caps_workers_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pseudobulk = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "perturbation": ["control", "gene_1", "control", "gene_1"],
                "cell_line": ["a", "a", "b", "b"],
            },
            index=["a_control", "a_gene_1", "b_control", "b_gene_1"],
        ),
    )
    observed: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, max_workers: int, mp_context: object) -> None:
            observed["max_workers"] = max_workers
            observed["mp_context"] = mp_context

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, function: object, *args: object) -> Future[list[dict[str, object]]]:
            future: Future[list[dict[str, object]]] = Future()
            context = args[1]
            n_cpus = args[-1]
            future.set_result([{"cell_line": context, "n_cpus": n_cpus}])
            return future

    monkeypatch.setattr(trade, "pseudobulk_replicates", lambda *args, **kwargs: pseudobulk)
    monkeypatch.setattr(trade, "ProcessPoolExecutor", FakeExecutor)

    result = perturbation_effect_statistics(
        pseudobulk,
        context_key="cell_line",
        context_workers=10,
    )

    assert observed["max_workers"] == 2
    assert result.to_dict("records") == [
        {"cell_line": "a", "n_cpus": 1},
        {"cell_line": "b", "n_cpus": 1},
    ]
    progress = capsys.readouterr().out
    assert "TRADE: context 1/2 complete" in progress
    assert "TRADE: context 2/2 complete" in progress
    assert "cell_line=a" in progress
    assert "cell_line=b" in progress


def test_context_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="context_workers must be >= 1"):
        perturbation_effect_statistics(ad.AnnData(X=np.ones((1, 1))), context_workers=0)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, None), ("", None), ("  ", None), ("32", 32)],
)
def test_deseq2_n_cpus_reads_env_override(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: int | None,
) -> None:
    monkeypatch.delenv("DESEQ2_N_CPUS", raising=False)
    if env_value is not None:
        monkeypatch.setenv("DESEQ2_N_CPUS", env_value)

    assert trade._deseq2_n_cpus() == expected


def test_sequential_path_forwards_n_cpus_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single-context path must honour DESEQ2_N_CPUS instead of hard-coding None."""
    pseudobulk = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"perturbation": ["control", "gene_1"], "cell_line": ["a", "a"]},
            index=["a_control", "a_gene_1"],
        ),
    )
    monkeypatch.setattr(trade, "pseudobulk_replicates", lambda *args, **kwargs: pseudobulk)
    monkeypatch.setattr(
        trade,
        "_perturbation_effect_rows_for_context",
        lambda sub, context, *args: [{"cell_line": context, "n_cpus": args[-1]}],
    )
    monkeypatch.setenv("DESEQ2_N_CPUS", "16")

    result = perturbation_effect_statistics(
        pseudobulk,
        context_key="cell_line",
        context_workers=1,
    )

    assert result.to_dict("records") == [{"cell_line": "a", "n_cpus": 16}]
