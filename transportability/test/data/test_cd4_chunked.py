"""Regression coverage for chunked CD4 embeddings and the STATE runner route."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbations.data.cd4_chunked import CD4ChunkedDataset

EMBEDDING_KEY = "X_geneformer"
EMBEDDINGS = np.arange(18, dtype=np.float32).reshape(6, 3)
EXPRESSION = np.arange(12, dtype=np.float32).reshape(6, 2)


def _write_manifest(
    directory: Path,
    *,
    dense_x: bool = False,
    sparse_embedding: bool = False,
    with_embedding: bool = True,
    second_embedding: np.ndarray | None = None,
) -> Path:
    """Write two tiny chunks in reverse manifest order to exercise canonical ordering."""
    chunks = []
    for chunk_id, donor in enumerate(("donor_a", "donor_b")):
        context = f"{donor}_Rest"
        row_slice = slice(3 * chunk_id, 3 * (chunk_id + 1))
        expression = EXPRESSION[row_slice].copy()
        chunk = ad.AnnData(
            X=expression if dense_x else sparse.csr_matrix(expression),
            obs=pd.DataFrame(
                {
                    "perturbation": ["control", "geneA", "geneB"],
                    "condition": ["control", "geneA", "geneB"],
                    "donor": [donor] * 3,
                    "timepoint": ["Rest"] * 3,
                    "context": [context] * 3,
                },
                index=[f"cell_{i}" for i in range(3)],
            ),
            var=pd.DataFrame(index=["geneA", "geneB"]),
        )
        chunk.layers["counts"] = sparse.csr_matrix(expression * 2)
        if with_embedding:
            embedding = (
                second_embedding
                if chunk_id == 1 and second_embedding is not None
                else EMBEDDINGS[row_slice].copy()
            )
            chunk.obsm[EMBEDDING_KEY] = (
                sparse.csr_matrix(embedding) if sparse_embedding else embedding
            )
        path = directory / f"chunk_{chunk_id}.h5ad"
        chunk.write_h5ad(path)
        chunks.append(
            {
                "path": path.name,
                "context": context,
                "donor": donor,
                "timepoint": "Rest",
                "chunk_index_within_context": 0,
                "n_obs": 3,
                "n_vars": 2,
            }
        )
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"chunks": chunks[::-1]}), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("dense_x", [False, True])
@pytest.mark.parametrize("sparse_embedding", [False, True])
def test_embeddings_keep_cell_order_across_chunks_and_disk(
    tmp_path: Path, dense_x: bool, sparse_embedding: bool
) -> None:
    manifest = _write_manifest(tmp_path, dense_x=dense_x, sparse_embedding=sparse_embedding)
    runtime = CD4ChunkedDataset.from_manifest(manifest, required_obsm_keys=(EMBEDDING_KEY,))
    indices = np.array([5, 1, 3, 0, 4, 2])
    output = tmp_path / "subset.h5ad"
    subset = runtime.materialize_subset(
        indices, "normalized_log1p", include_layers=("counts",), output_path=output
    )
    for materialized in (subset, ad.read_h5ad(output)):
        assert materialized.obs_names.tolist() == runtime.obs.index[indices].tolist()
        np.testing.assert_array_equal(materialized.obsm[EMBEDDING_KEY], EMBEDDINGS[indices])
        assert materialized.obsm[EMBEDDING_KEY].dtype == np.float32
        x = materialized.X.toarray() if sparse.issparse(materialized.X) else materialized.X
        np.testing.assert_array_equal(x, EXPRESSION[indices])
        np.testing.assert_array_equal(
            materialized.layers["counts"].toarray(), 2 * EXPRESSION[indices]
        )
        assert "donor_timepoint" in materialized.obs


def test_empty_subset_keeps_embedding_width_and_layers(tmp_path: Path) -> None:
    runtime = CD4ChunkedDataset.from_manifest(
        _write_manifest(tmp_path), required_obsm_keys=(EMBEDDING_KEY,)
    )
    output = tmp_path / "empty.h5ad"
    runtime.materialize_subset(
        np.array([], dtype=np.int64), None, include_layers=("counts",), output_path=output
    )
    empty = ad.read_h5ad(output)
    assert empty.shape == (0, 2)
    assert empty.obsm[EMBEDDING_KEY].shape == (0, 3)
    assert empty.obsm[EMBEDDING_KEY].dtype == np.float32
    assert empty.layers["counts"].shape == (0, 2)


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        (np.zeros((3, 4), dtype=np.float32), "width 4; expected 3"),
        (np.full((3, 3), np.nan), "non-finite"),
        (np.full((3, 3), np.inf), "non-finite"),
        (np.zeros((3, 0), dtype=np.float32), "positive width"),
        (np.full((3, 3), 1e40), "float32 range"),
        (np.full((3, 3), "bad", dtype=object), "real numeric"),
    ],
)
def test_invalid_embedding_in_later_chunk_fails_before_materialization(
    tmp_path: Path, embedding: np.ndarray, message: str
) -> None:
    manifest = _write_manifest(tmp_path, second_embedding=embedding)
    with pytest.raises(ValueError, match=message):
        CD4ChunkedDataset.from_manifest(manifest, required_obsm_keys=(EMBEDDING_KEY,))


def test_missing_embedding_in_later_chunk_is_not_silently_dropped(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    second_path = tmp_path / "chunk_1.h5ad"
    second = ad.read_h5ad(second_path)
    del second.obsm[EMBEDDING_KEY]
    second.write_h5ad(second_path)
    with pytest.raises(KeyError, match=r"chunk_1.*X_geneformer.*missing"):
        CD4ChunkedDataset.from_manifest(manifest, required_obsm_keys=(EMBEDDING_KEY,))


def test_manifest_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["chunks"][0]["n_obs"] = 4
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"manifest n_obs=4.*chunk n_obs=3"):
        CD4ChunkedDataset.from_manifest(manifest, required_obsm_keys=(EMBEDDING_KEY,))


def test_expression_only_runtime_remains_supported(tmp_path: Path) -> None:
    runtime = CD4ChunkedDataset.from_manifest(_write_manifest(tmp_path, with_embedding=False))
    subset = runtime.materialize_subset(np.array([4, 1]), None)
    assert not subset.obsm
    np.testing.assert_array_equal(subset.X.toarray(), EXPRESSION[[4, 1]])


@pytest.mark.parametrize("strategy", ["in-context", "cross-context"])
def test_json_runner_delivers_embeddings_and_seed_to_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    from perturbations.analyses.real_experiments import run

    manifest = _write_manifest(tmp_path)
    split_indices = (np.array([3, 1, 0]), np.array([4]), np.array([5, 2]))
    cache_root = tmp_path / "split_cache"
    state_root = tmp_path / "state" / strategy
    monkeypatch.setenv("CD4_SPLIT_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("STATE_OUTPUT_ROOT", str(state_root))
    state_calls = []

    def prepare_from_cache(*, split_paths, split_metadata):
        test = ad.read_h5ad(split_paths["test"])
        return SimpleNamespace(test_obs=test.obs.copy(), test_var=test.var.copy())

    def call_state(**kwargs):
        state_calls.append(kwargs)
        assert kwargs["basal_embedding_key"] == EMBEDDING_KEY
        assert kwargs["seed"] == 7
        assert kwargs["model_dir"] == str(state_root / "trial_7")
        assert kwargs["checkpoint_dir"] == str(state_root / "trial_7" / "checkpoints")
        assert kwargs["resume_from_checkpoint"] == "last"
        for name, indices in zip(
            ("train_adata", "valid_adata", "test_adata"), split_indices, strict=True
        ):
            np.testing.assert_array_equal(kwargs[name].obsm[EMBEDDING_KEY], EMBEDDINGS[indices])
            np.testing.assert_array_equal(kwargs[name].X.toarray(), EXPRESSION[indices])
        return {"preds": np.ones((2, 2), dtype=np.float32)}

    def run_selected_models(**kwargs):
        assert kwargs["models"] == ("STATE",)
        prediction = kwargs["run_state"]()
        np.testing.assert_array_equal(prediction.X, np.ones((2, 2), dtype=np.float32))
        return [{"model": "STATE", "trial_id": 7, "status": "success"}]

    def run_runtime(**kwargs):
        assert kwargs["split_strategy"] == strategy
        assert kwargs["basal_embedding_key"] == EMBEDDING_KEY
        assert kwargs["splitter_adata"].obsm_widths == {EMBEDDING_KEY: 3}
        splitter = SimpleNamespace(
            split=lambda seed: split_indices,
            get_split_metadata=lambda: SimpleNamespace(context_axis="donor_timepoint"),
        )
        rows = kwargs["run_trial"](splitter, 7)
        assert rows == [{"model": "STATE", "trial_id": 7, "status": "success"}]
        return "results.csv"

    monkeypatch.setattr(run, "_prepare_cd4_trial_data_from_cache", prepare_from_cache)
    monkeypatch.setattr(run, "run_state_gene", call_state)
    monkeypatch.setattr(run, "_run_trial_models", run_selected_models)
    monkeypatch.setattr(run, "_run_real_experiments_with_runtime", run_runtime)
    result = run.run_real_experiments(
        dataset_name="CD4+",
        dataset_path=str(manifest),
        output_dir=str(tmp_path / "results"),
        n_trials=10,
        trial_ids=(7,),
        counts_layer="counts",
        obs_layer="normalized_log1p",
        split_strategy=strategy,
        norm_target_sum=1e4,
        basal_embedding_key=EMBEDDING_KEY,
        models=("STATE",),
    )
    assert result == "results.csv"
    assert len(state_calls) == 1
    assert cache_root.is_dir()
    assert list(cache_root.iterdir()) == []


def test_csv_rows_record_embedding_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from perturbations.analyses.real_experiments import run

    monkeypatch.setattr(run, "_release_process_memory", lambda: None)
    summary = run.DatasetSummary(
        dataset="CD4+",
        dataset_variant=None,
        dataset_path="manifest.json",
        n_cells=6,
        n_genes=2,
        n_cell_lines=2,
        n_total_perturbations=2,
        sparsity=0.1,
        basal_embedding_key=EMBEDDING_KEY,
    )
    rows = run._execute_trial_loop(
        trial_ids=(7,),
        models=("STATE",),
        splitter=None,
        summary=summary,
        run_trial=lambda seed: [{"trial_id": seed, "model": "STATE", "status": "success"}],
    )
    output = tmp_path / "results.csv"
    run._write_results(
        all_rows=rows,
        csv_file=output,
        error_log_file=tmp_path / "errors.txt",
        trial_ids=(7,),
        models=("STATE",),
    )
    result = pd.read_csv(output)
    assert result["basal_embedding_key"].tolist() == [EMBEDDING_KEY]
    assert result["model"].tolist() == ["STATE"]
