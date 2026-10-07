"""Small synthetic extraction checks; execute only within a Slurm allocation."""

from __future__ import annotations

import json
import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import extract_cd4_geneformer as cd4


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_dir = tmp_path / "processed"
    source_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"hidden_size": 2}')
    matrix = np.array([[1, 2, 3, 4], [2, 0, 2, 0], [4, 3, 2, 1], [0, 0, 0, 3], [2, 4, 1, 2]], dtype=np.float32)
    raw = ad.AnnData(
        X=sparse.csr_matrix(matrix),
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c", "cell_d", "cell_e"]),
        var=pd.DataFrame({"gene_ids": ["ENSG000001", "ENSG000002", "ENSG000003", "ENSG000004"]}, index=["g1", "g2", "g3", "g4"]),
    )
    raw.write_h5ad(raw_dir / "D1_Rest.assigned_guide.h5ad")
    # Deliberately unsorted raw positions must still yield exact processed order.
    retained = [4, 0, 2]
    processed = raw[retained, :2].copy()
    processed.layers["counts"] = processed.X.copy()
    processed.X.data = np.log1p(processed.X.data)
    processed.obs["context"] = "D1_Rest"
    processed.obs["donor"] = "D1"
    processed.obs["timepoint"] = "Rest"
    processed.obs["condition"] = pd.Categorical(["control", "ENSG000001", "control"])
    processed.uns["log1p"] = {"base": None}
    processed.obsm["other_embedding"] = np.arange(6).reshape(3, 2)
    processed.write_h5ad(source_dir / "chunk.h5ad")
    manifest = {"n_obs_total": 3, "n_vars_final": 2, "chunks": [{"path": "chunk.h5ad", "context": "D1_Rest", "donor": "D1", "timepoint": "Rest", "chunk_index_within_context": 0, "n_obs": 3, "n_vars": 2}]}
    manifest_path = source_dir / "processed_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = cd4.parse_args(["--manifest", str(manifest_path), "--raw-dir", str(raw_dir), "--model-dir", str(model_dir), "--output-dir", str(tmp_path / "output"), "--work-dir", str(tmp_path / "work"), "--chunk-index", "0", "--batch-size", "2"])
    calls = []

    def fake_extract(batch, **kwargs):
        assert kwargs["counts_layer"] is None
        assert kwargs["ensembl_col"] == "gene_ids"
        assert batch.n_vars == 4
        values = batch.X.toarray()
        calls.append(batch.obs_names.tolist())
        batch.obsm[kwargs["obsm_key"]] = np.column_stack([values.sum(axis=1), values[:, 3]])
        return batch

    monkeypatch.setattr(cd4, "extract_geneformer_embeddings", fake_extract)
    source_manifest, entries = cd4._load_manifest(args.manifest)
    configuration = cd4._run_configuration(args)
    return args, source_manifest, entries, configuration, calls, matrix[retained]


def test_batch_resume_preserves_cells_full_gene_input_and_processed_data(dataset):
    args, manifest, entries, configuration, calls, raw_values = dataset
    args.max_batches = 1
    assert not cd4.extract_chunk(args, entries[0], configuration)
    assert len(calls) == 1
    assert not (args.output_dir / "processed_manifest.json").exists()
    assert not list((args.output_dir / "chunks").glob("*.h5ad"))
    args.max_batches = None
    assert cd4.extract_chunk(args, entries[0], configuration)
    assert len(calls) == 2  # Completed pilot batch was reused.
    result = cd4.finalize(args, manifest, entries, configuration)
    enriched_manifest = json.loads(result.read_text())
    output = args.output_dir / enriched_manifest["chunks"][0]["path"]
    original = ad.read_h5ad(entries[0]["path"])
    enriched = ad.read_h5ad(output)
    pd.testing.assert_frame_equal(enriched.obs, original.obs)
    pd.testing.assert_frame_equal(enriched.var, original.var)
    np.testing.assert_array_equal(enriched.X.toarray(), original.X.toarray())
    np.testing.assert_array_equal(enriched.layers["counts"].toarray(), original.layers["counts"].toarray())
    np.testing.assert_array_equal(enriched.obsm["other_embedding"], original.obsm["other_embedding"])
    np.testing.assert_array_equal(enriched.obsm["X_geneformer"], np.column_stack([raw_values.sum(axis=1), raw_values[:, 3]]))
    assert cd4.extract_chunk(args, entries[0], configuration)
    assert len(calls) == 2  # Completed output is verified and reused.
    assert enriched_manifest["geneformer"]["complete"]


def test_incomplete_dataset_never_publishes_manifest(dataset):
    args, manifest, entries, configuration, _, _ = dataset
    with pytest.raises(FileNotFoundError, match="incomplete"):
        cd4.finalize(args, manifest, entries, configuration)
    assert not (args.output_dir / "processed_manifest.json").exists()


def test_changed_config_refuses_existing_checkpoints(dataset):
    args, _, entries, configuration, _, _ = dataset
    args.max_batches = 1
    cd4.extract_chunk(args, entries[0], configuration)
    args.forward_batch_size = 1
    with pytest.raises(ValueError, match="changed"):
        cd4.extract_chunk(args, entries[0], cd4._run_configuration(args))


def test_corrupted_checkpoint_refuses_reuse(dataset):
    args, _, entries, configuration, _, _ = dataset
    args.max_batches = 1
    cd4.extract_chunk(args, entries[0], configuration)
    checkpoint = next(args.work_dir.glob("*/batch_*.npz"))
    with np.load(checkpoint, allow_pickle=False) as values:
        identity = json.loads(str(values["identity"].item()))
    identity["obs_sha256"] = "wrong_order"
    cd4._checkpoint(checkpoint, np.zeros((2, 2), dtype=np.float32), identity, 1)
    with pytest.raises(ValueError, match="Checkpoint provenance"):
        cd4.extract_chunk(args, entries[0], configuration)


def test_finalization_detects_tampered_embeddings(dataset):
    args, manifest, entries, configuration, _, _ = dataset
    cd4.extract_chunk(args, entries[0], configuration)
    _, output, _ = cd4._chunk_paths(args, entries[0])
    with h5py.File(output, "r+") as handle:
        handle["obsm"]["X_geneformer"][0, 0] += 1
    with pytest.raises(ValueError, match="completed receipt"):
        cd4.finalize(args, manifest, entries, configuration)
    assert not (args.output_dir / "processed_manifest.json").exists()


def test_manifest_and_receipts_survive_directory_move(dataset, tmp_path):
    args, manifest, entries, configuration, _, _ = dataset
    cd4.extract_chunk(args, entries[0], configuration)
    cd4.finalize(args, manifest, entries, configuration)
    moved = tmp_path / "persistent"
    shutil.copytree(args.output_dir, moved)
    args.output_dir = moved
    path = cd4.finalize(args, manifest, entries, configuration)
    assert (moved / json.loads(path.read_text())["chunks"][0]["path"]).exists()


@pytest.mark.parametrize("processed,raw", [(["a", "a"], ["a", "b"]), (["a"], ["a", "a"]), (["missing"], ["a", "b"])])
def test_alignment_rejects_ambiguous_or_missing_cells(processed, raw):
    with pytest.raises(ValueError):
        cd4._align_rows(pd.Index(processed), pd.Index(raw))


@pytest.mark.parametrize("values", [[[1.2, 2]], [[-1, 2]], [[np.nan, 2]], [[np.inf, 2]], [[0, 0]]])
def test_raw_count_validation_rejects_invalid_inputs(values):
    with pytest.raises(ValueError):
        cd4._validate_counts(sparse.csr_matrix(values))


def test_extraction_refuses_hvg_input_as_raw(dataset):
    args, _, entries, configuration, _, _ = dataset
    source = ad.read_h5ad(entries[0]["path"])
    source.write_h5ad(args.raw_dir / "D1_Rest.assigned_guide.h5ad")
    with pytest.raises(ValueError, match="more genes"):
        cd4.extract_chunk(args, entries[0], configuration)


def test_max_cells_stops_only_at_complete_batch_boundaries(dataset):
    args, _, entries, configuration, calls, _ = dataset
    args.max_cells = 2
    assert not cd4.extract_chunk(args, entries[0], configuration)
    assert len(calls) == 1
    args.max_cells = None
    assert cd4.extract_chunk(args, entries[0], configuration)
    assert len(calls) == 2


def test_sparse_csc_input_fails_before_large_materialization(tmp_path):
    path = tmp_path / "csc.h5ad"
    ad.AnnData(X=sparse.csc_matrix([[1, 2], [3, 4]])).write_h5ad(path)
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="must be CSR"):
        cd4._read_count_batch(handle, np.array([1, 0]))


def test_dense_count_slicing_preserves_requested_order(tmp_path):
    path = tmp_path / "dense.h5ad"
    ad.AnnData(X=np.array([[1, 2], [3, 4], [5, 6]])).write_h5ad(path)
    with h5py.File(path, "r") as handle:
        counts = cd4._read_count_batch(handle, np.array([2, 0]))
    np.testing.assert_array_equal(counts, [[5, 6], [1, 2]])
