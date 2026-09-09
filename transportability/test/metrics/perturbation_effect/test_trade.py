from concurrent.futures import Future

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

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


def _synthetic_pseudobulk(
    n_conditions: int = 4,
    n_replicates: int = 3,
    n_genes: int = 25,
    n_batches: int = 2,
    seed: int = 0,
) -> ad.AnnData:
    """Build a small pseudobulk AnnData with a batch structure DESeq2 can fit."""
    rng = np.random.default_rng(seed)
    labels: list[str] = []
    batches: list[str] = []
    for condition in range(n_conditions):
        name = "control" if condition == 0 else f"gene_{condition}"
        for replicate in range(n_replicates):
            labels.append(name)
            batches.append(f"batch_{replicate % n_batches}")

    counts = rng.negative_binomial(20, 0.3, size=(len(labels), n_genes)).astype(np.int64)
    # Give a couple of genes a real, condition-specific effect.
    for condition in range(1, n_conditions):
        mask = np.asarray(labels) == f"gene_{condition}"
        counts[mask, condition] += 60

    return ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"perturbation": labels, "batch": batches},
            index=[f"s{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_deseq2_effect_sizes_matches_per_contrast_deseqstats(batch_key: str | None) -> None:
    """
    The hoisted Wald path must reproduce PyDESeq2's one-contrast-at-a-time results.

    ``deseq2_effect_sizes`` computes the per-gene covariance matrix and its
    inverse once and reuses them across contrasts. This pins that optimization
    to the reference implementation it replaced: a fresh ``DeseqStats.summary()``
    per contrast, which recomputes both every time.
    """
    pytest.importorskip("pydeseq2")

    pseudobulk = _synthetic_pseudobulk()
    optimized = trade.deseq2_effect_sizes(pseudobulk, batch_key=batch_key, quiet=True, n_cpus=1)

    # Reference: rebuild the fit and run an independent DeseqStats per contrast.
    labels = np.asarray(pseudobulk.obs["perturbation"]).astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    safe = {lab: f"c{i}" for i, lab in enumerate(unique_labels)}
    metadata = pd.DataFrame(
        {"condition": [safe[lab] for lab in labels]},
        index=np.asarray(pseudobulk.obs_names, dtype=str),
    )
    design = "~condition"
    if batch_key is not None:
        batch_values = np.asarray(pseudobulk.obs[batch_key]).astype(str)
        safe_batch = {b: f"b{i}" for i, b in enumerate(dict.fromkeys(batch_values.tolist()))}
        metadata["batch"] = [safe_batch[b] for b in batch_values]
        design = "~batch + condition"

    dds = DeseqDataSet(
        counts=pd.DataFrame(
            np.asarray(pseudobulk.X, dtype=np.int64),
            index=np.asarray(pseudobulk.obs_names, dtype=str),
            columns=np.asarray(pseudobulk.var_names, dtype=str),
        ),
        metadata=metadata,
        design=design,
        ref_level=["condition", safe["control"]],
        quiet=True,
        n_cpus=1,
    )
    dds.deseq2()

    assert set(optimized) == {lab for lab in unique_labels if lab != "control"}
    for lab in optimized:
        stats = DeseqStats(
            dds,
            contrast=["condition", safe[lab], safe["control"]],
            quiet=True,
            n_cpus=1,
        )
        stats.summary()
        expected = stats.results_df
        actual = optimized[lab]
        for column in ["log2FoldChange", "lfcSE", "pvalue", "padj"]:
            np.testing.assert_allclose(
                actual[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"{column} diverged for contrast {lab!r}",
                equal_nan=True,
            )


def test_wald_tests_all_contrasts_rejects_mismatched_contrast_length() -> None:
    pytest.importorskip("pydeseq2")
    from pydeseq2.dds import DeseqDataSet

    pseudobulk = _synthetic_pseudobulk(n_conditions=2, n_genes=8)
    labels = np.asarray(pseudobulk.obs["perturbation"]).astype(str)
    metadata = pd.DataFrame(
        {"condition": ["c0" if lab == "control" else "c1" for lab in labels]},
        index=np.asarray(pseudobulk.obs_names, dtype=str),
    )
    dds = DeseqDataSet(
        counts=pd.DataFrame(
            np.asarray(pseudobulk.X, dtype=np.int64),
            index=np.asarray(pseudobulk.obs_names, dtype=str),
            columns=np.asarray(pseudobulk.var_names, dtype=str),
        ),
        metadata=metadata,
        design="~condition",
        ref_level=["condition", "c0"],
        quiet=True,
        n_cpus=1,
    )
    dds.deseq2()

    with pytest.raises(ValueError, match="Contrast vectors have"):
        trade._wald_tests_all_contrasts(dds, np.ones((99, 1)), n_cpus=1)


def test_wald_tests_all_contrasts_parallel_matches_serial() -> None:
    """
    The chunked multi-worker path must agree with the single-chunk path.

    ``_wald_tests_all_contrasts`` splits genes across workers, so the production
    path evaluates ``X @ LFC_chunk.T`` per chunk while the serial path does one
    ``X @ LFC.T``. BLAS selects different kernels by matrix shape, so these agree
    only to within floating-point tolerance, not bit-for-bit. Validated on real
    data: reruns of the Replogle22 HepG2 pair reproduced the reference results to
    a maximum relative deviation of ~1e-13, with DEG counts and mean |LFC|
    bit-identical.

    The n_cpus=1 equivalence test above exercises only the single-chunk branch,
    so this pins the branch that actually runs in production.
    """
    pytest.importorskip("pydeseq2")
    from pydeseq2.dds import DeseqDataSet

    pseudobulk = _synthetic_pseudobulk(n_conditions=6, n_genes=60)
    labels = np.asarray(pseudobulk.obs["perturbation"]).astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    safe = {lab: f"c{i}" for i, lab in enumerate(unique_labels)}
    metadata = pd.DataFrame(
        {"condition": [safe[lab] for lab in labels]},
        index=np.asarray(pseudobulk.obs_names, dtype=str),
    )
    dds = DeseqDataSet(
        counts=pd.DataFrame(
            np.asarray(pseudobulk.X, dtype=np.int64),
            index=np.asarray(pseudobulk.obs_names, dtype=str),
            columns=np.asarray(pseudobulk.var_names, dtype=str),
        ),
        metadata=metadata,
        design="~condition",
        quiet=True,
        n_cpus=1,
    )
    dds.deseq2()

    contrasts = np.column_stack(
        [
            np.asarray(
                dds.contrast(
                    column="condition", baseline=safe["control"], group_to_compare=safe[lab]
                ),
                dtype=np.float64,
            )
            for lab in unique_labels
            if lab != "control"
        ]
    )

    se_serial, stat_serial, pval_serial = trade._wald_tests_all_contrasts(dds, contrasts, n_cpus=1)
    se_par, stat_par, pval_par = trade._wald_tests_all_contrasts(dds, contrasts, n_cpus=4)

    for name, a, b in (
        ("se", se_serial, se_par),
        ("stat", stat_serial, stat_par),
        ("pvalue", pval_serial, pval_par),
    ):
        np.testing.assert_allclose(
            a, b, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f"{name} diverged"
        )


def test_size_factors_fit_type_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DESEQ2_SIZE_FACTORS_FIT_TYPE`` must be read, validated, and default to unset."""
    monkeypatch.delenv("DESEQ2_SIZE_FACTORS_FIT_TYPE", raising=False)
    assert trade._deseq2_size_factors_fit_type() is None

    for value in ("ratio", "poscounts", "iterative"):
        monkeypatch.setenv("DESEQ2_SIZE_FACTORS_FIT_TYPE", value)
        assert trade._deseq2_size_factors_fit_type() == value

    monkeypatch.setenv("DESEQ2_SIZE_FACTORS_FIT_TYPE", "bogus")
    with pytest.raises(ValueError, match="must be one of"):
        trade._deseq2_size_factors_fit_type()


def test_poscounts_avoids_iterative_fallback_on_sparse_pseudobulk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On pseudobulk where every gene has a zero, poscounts must skip the iterative path.

    PyDESeq2 falls back to ``_fit_iterate_size_factors`` when no gene is non-zero
    in every sample. That fallback runs derivative-free Powell over one parameter
    per sample and is intractable on a real panel, so this pins the escape hatch:
    with the override set, the fallback must not be reached.
    """
    pytest.importorskip("pydeseq2")
    from pydeseq2.dds import DeseqDataSet

    pseudobulk = _synthetic_pseudobulk(n_conditions=4, n_genes=30)
    counts = np.asarray(pseudobulk.X)
    # Force the trigger: give every gene at least one all-zero sample.
    for gene in range(counts.shape[1]):
        counts[gene % counts.shape[0], gene] = 0
    pseudobulk.X = counts
    assert (counts == 0).any(axis=0).all(), "test setup must trigger the fallback"

    called: dict[str, bool] = {"iterative": False}

    def _boom(self: DeseqDataSet, *args: object, **kwargs: object) -> None:
        called["iterative"] = True
        raise AssertionError("iterative size-factor fallback was reached")

    monkeypatch.setattr(DeseqDataSet, "_fit_iterate_size_factors", _boom)
    monkeypatch.setenv("DESEQ2_SIZE_FACTORS_FIT_TYPE", "poscounts")

    effects = trade.deseq2_effect_sizes(pseudobulk, batch_key=None, quiet=True, n_cpus=1)

    assert not called["iterative"]
    assert len(effects) == 3
