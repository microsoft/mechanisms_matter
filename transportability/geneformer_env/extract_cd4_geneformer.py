"""Embed retained CD4 cells from full-gene raw counts, with resumable batches.

Run this script inside a Slurm allocation using the isolated Geneformer environment.
Use one array task per manifest entry (zero-based ``--chunk-index``), then run
``--finalize`` with the same arguments to validate all chunks and publish a new
``processed_manifest.json``. No manifest is written by extraction or pilot runs.
All generated files, caches and TMPDIR must be on scratch or SLURM_TMPDIR.

``--max-batches 1`` is a pilot that leaves a reusable production-size checkpoint.
``--max-cells`` caps work at complete batch boundaries, never splitting a batch.
Do not change batch size, model files, or extraction settings while resuming.
Raw .X contains UMI counts according to data/cd4+/README.md; every input batch
is additionally checked for finite, nonnegative integer counts before inference.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata.io import read_elem, sparse_dataset, write_elem
except ImportError:  # anndata < 0.11
    from anndata._core.sparse_dataset import sparse_dataset
    from anndata._io.specs import read_elem, write_elem

from extract_geneformer_embeddings import extract_geneformer_embeddings

SCHEMA_VERSION = 1
PROVENANCE_KEY = "geneformer_extraction_json"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _index_hash(index: pd.Index) -> str:
    # Length-delimited JSON cannot confuse embedded separators with cell boundaries.
    return _json_hash(index.astype(str).tolist())


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_index(group: h5py.Group) -> pd.Index:
    key = group.attrs.get("_index", "_index")
    return pd.Index(read_elem(group[key]).astype(str))


def _resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else manifest.parent / path).resolve(strict=True)


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    entries = copy.deepcopy(manifest.get("chunks") or manifest.get("contexts") or [])
    if not entries:
        raise ValueError("Source manifest has no chunks or contexts.")
    keys = set()
    for entry in entries:
        entry["path"] = str(_resolve_path(path, entry["path"]))
        entry.setdefault("chunk_index_within_context", 0)
        context = str(entry["context"])
        if not re.fullmatch(r"D\d+_(Rest|Stim8hr|Stim48hr)", context):
            raise ValueError(f"Unexpected CD4 context: {context!r}")
        key = (context, int(entry["chunk_index_within_context"]))
        if key in keys:
            raise ValueError(f"Duplicate manifest chunk: {key}")
        keys.add(key)
        if int(entry["n_obs"]) <= 0 or int(entry["n_vars"]) <= 0:
            raise ValueError("Manifest chunks must have positive dimensions.")
    if len({entry["n_vars"] for entry in entries}) != 1:
        raise ValueError("Manifest chunks disagree on the processed gene count.")
    if "n_obs_total" in manifest and sum(int(e["n_obs"]) for e in entries) != int(manifest["n_obs_total"]):
        raise ValueError("Source manifest n_obs_total disagrees with its chunks.")
    return manifest, entries


def _run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model_dir.resolve(strict=True)
    candidates = [model.parent / "geneformer", model, model.parent]
    if args.dictionary_dir:
        candidates.insert(0, args.dictionary_dir.resolve(strict=True))
    assets = set(model.rglob("*.json")) | set(model.rglob("*.safetensors")) | set(model.rglob("*.bin"))
    for candidate in candidates:
        if candidate.is_dir():
            assets.update(candidate.glob("*.pkl"))
    if not assets:
        raise ValueError(f"No model assets found under {model}.")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": _file_identity(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "model_dir": str(model),
        "dictionary_dir": str(args.dictionary_dir.resolve()) if args.dictionary_dir else None,
        "assets": [_file_identity(path) for path in sorted(assets)],
        "counts_source": "raw.X",
        "obsm_key": args.obsm_key,
        "batch_size": args.batch_size,
        "model_input_size": args.model_input_size,
        "forward_batch_size": args.forward_batch_size,
        "emb_mode": "cls",
        "special_token": True,
    }


def _chunk_paths(args: argparse.Namespace, entry: dict[str, Any]) -> tuple[Path, Path, Path]:
    stem = f"{entry['context']}_chunk{int(entry['chunk_index_within_context']):03d}"
    return (
        args.work_dir / stem,
        args.output_dir / "chunks" / f"cd4_geneformer_{stem}.h5ad",
        args.output_dir / "receipts" / f"{stem}.json",
    )


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"An extraction process already holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _align_rows(processed: pd.Index, raw: pd.Index) -> np.ndarray:
    if not processed.is_unique or not raw.is_unique:
        raise ValueError("Cell names must be unique within each processed and raw context.")
    positions = raw.get_indexer(processed)
    if np.any(positions < 0):
        raise ValueError(f"{int((positions < 0).sum())} retained cells are missing from their raw context.")
    if not raw.take(positions).equals(processed):
        raise ValueError("Exact raw-to-processed cell alignment failed.")
    return positions


def _validate_counts(counts: Any) -> None:
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Raw .X must contain finite, nonnegative UMI counts.")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("Raw .X contains noninteger values; normalized expression is not a valid input.")
    if np.any(np.asarray(counts.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("Raw .X contains retained cells with zero library size.")


def _read_count_batch(raw: h5py.File, positions: np.ndarray) -> Any:
    if isinstance(raw["X"], h5py.Group) and raw["X"].attrs.get("encoding-type") != "csr_matrix":
        raise ValueError("Raw sparse .X must be CSR for bounded row extraction; CSC requires a separate conversion job.")
    matrix = sparse_dataset(raw["X"]) if isinstance(raw["X"], h5py.Group) else raw["X"]
    order = np.argsort(positions)
    counts = matrix[positions[order], :][np.argsort(order), :]
    _validate_counts(counts)
    return counts


def _validate_embedding(embedding: np.ndarray, n_cells: int, dimension: int | None = None) -> None:
    if embedding.ndim != 2 or embedding.shape[0] != n_cells or embedding.shape[1] < 1:
        raise ValueError(f"Invalid embedding shape {embedding.shape} for {n_cells} cells.")
    if dimension is not None and embedding.shape[1] != dimension:
        raise ValueError("Embedding dimensions changed between batches or chunks.")
    if not np.isfinite(embedding).all():
        raise ValueError("Embeddings contain nonfinite values.")


def _checkpoint(path: Path, embedding: np.ndarray, identity: dict[str, Any], seconds: float) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, embeddings=embedding, identity=json.dumps(identity, sort_keys=True), seconds=seconds)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path, expected: dict[str, Any]) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["identity"].item())) != expected:
            raise ValueError(f"Checkpoint provenance does not match: {path}")
        embedding = np.asarray(data["embeddings"], dtype=np.float32)
        seconds = float(data["seconds"])
    _validate_embedding(embedding, expected["end"] - expected["start"])
    return embedding, seconds


def _embedding_digest(dataset: h5py.Dataset, batch_size: int = 4096) -> str:
    digest = hashlib.sha256()
    for start in range(0, dataset.shape[0], batch_size):
        values = np.asarray(dataset[start:start + batch_size], dtype=np.float32)
        _validate_embedding(values, len(values))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_output(path: Path, provenance: dict[str, Any], receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        stored = json.loads(str(read_elem(handle["uns"][PROVENANCE_KEY])))
        if stored != provenance:
            raise ValueError(f"Output provenance mismatch: {path}")
        if _index_hash(_read_index(handle["obs"])) != provenance["obs_sha256"]:
            raise ValueError("Output cell identities or order changed.")
        if _index_hash(_read_index(handle["var"])) != provenance["var_sha256"]:
            raise ValueError("Output processed gene identities or order changed.")
        embedding = handle["obsm"][provenance["configuration"]["obsm_key"]]
        if embedding.ndim != 2 or embedding.shape[0] != provenance["n_obs"] or embedding.shape[1] < 1:
            raise ValueError("Output embedding dimensions do not match its cells.")
        digest = _embedding_digest(embedding)
        dimension = int(embedding.shape[1])
    result = {
        "provenance": provenance,
        "embedding_dimension": dimension,
        "embeddings_sha256": digest,
        # Content verification plus a relative filename keeps receipts portable
        # when the completed directory is copied from scratch to project storage.
        "output": {"filename": path.name, "size": path.stat().st_size},
        "complete": True,
    }
    if receipt is not None and result != receipt:
        raise ValueError(f"Output does not match its completed receipt: {path}")
    return result


def _chunk_provenance(args: argparse.Namespace, entry: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    source = Path(entry["path"])
    raw = args.raw_dir / f"{entry['context']}.assigned_guide.h5ad"
    with h5py.File(source, "r") as handle:
        obs = _read_index(handle["obs"])
        var = _read_index(handle["var"])
        if len(obs) != int(entry["n_obs"]) or len(var) != int(entry["n_vars"]):
            raise ValueError("Processed chunk dimensions disagree with manifest.")
        if not obs.is_unique:
            raise ValueError("Processed cell names must be unique.")
        for column, expected in (("context", entry["context"]), ("donor", entry["donor"]), ("timepoint", entry["timepoint"])):
            if column not in handle["obs"] or not np.all(np.asarray(read_elem(handle["obs"][column])).astype(str) == str(expected)):
                raise ValueError(f"Processed obs[{column!r}] disagrees with manifest context.")
    return {
        "configuration": configuration,
        "configuration_sha256": _json_hash(configuration),
        "source": _file_identity(source),
        "raw_source": _file_identity(raw),
        "context": entry["context"],
        "chunk_index_within_context": int(entry["chunk_index_within_context"]),
        "n_obs": len(obs),
        "n_vars": len(var),
        "obs_sha256": _index_hash(obs),
        "var_sha256": _index_hash(var),
    }


def extract_chunk(args: argparse.Namespace, entry: dict[str, Any], configuration: dict[str, Any]) -> bool:
    work, output, receipt_path = _chunk_paths(args, entry)
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(work / "extraction.lock"):
        provenance = _chunk_provenance(args, entry, configuration)
        provenance_path = work / "provenance.json"
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Source/model/configuration changed; choose a new work directory.")
        _atomic_json(provenance_path, provenance)
        if output.exists():
            existing_receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            result = _validate_output(output, provenance, existing_receipt)
            _atomic_json(receipt_path, result)
            print(json.dumps({"event": "chunk_already_complete", "context": entry["context"]}), flush=True)
            return True

        with h5py.File(entry["path"], "r") as processed, h5py.File(provenance["raw_source"]["path"], "r") as raw:
            raw_x = raw["X"]
            encoding = str(raw_x.attrs.get("encoding-type", "array"))
            if isinstance(raw_x, h5py.Group) and encoding != "csr_matrix":
                raise ValueError("Raw sparse .X must be CSR for bounded row extraction; CSC requires a separate conversion job.")
            raw_shape = raw_x.attrs["shape"] if isinstance(raw_x, h5py.Group) else raw_x.shape
            print(json.dumps({"event": "raw_matrix", "context": entry["context"], "encoding": encoding, "shape": [int(value) for value in raw_shape]}), flush=True)
            cells = _read_index(processed["obs"])
            raw_cells = _read_index(raw["obs"])
            positions = _align_rows(cells, raw_cells)
            raw_var = read_elem(raw["var"])
            if len(raw_var) <= int(entry["n_vars"]):
                raise ValueError("Raw input does not contain more genes than the processed HVG matrix.")
            if "gene_ids" not in raw_var:
                raise ValueError("Expected authoritative Ensembl IDs in raw.var['gene_ids'].")
            n_batches = math.ceil(len(cells) / args.batch_size)
            if args.max_cells is not None and args.max_cells < min(args.batch_size, len(cells)):
                raise ValueError("--max-cells must allow at least one complete batch; decrease --batch-size for a smaller pilot.")
            complete = []
            new_batches = 0
            elapsed = 0.0
            embedded_cells = 0
            dimension = None
            for batch_index, start in enumerate(range(0, len(cells), args.batch_size)):
                end = min(start + args.batch_size, len(cells))
                identity = {"provenance_sha256": _json_hash(provenance), "start": start, "end": end, "obs_sha256": _index_hash(cells[start:end])}
                checkpoint_path = work / f"batch_{batch_index:06d}.npz"
                if checkpoint_path.exists():
                    embedding, seconds = _load_checkpoint(checkpoint_path, identity)
                else:
                    if (args.max_batches is not None and new_batches >= args.max_batches) or (args.max_cells is not None and end > args.max_cells):
                        break
                    started = time.monotonic()
                    counts = _read_count_batch(raw, positions[start:end])
                    batch = ad.AnnData(X=counts, obs=pd.DataFrame(index=cells[start:end]), var=raw_var.copy())
                    extract_geneformer_embeddings(
                        batch, model_dir=str(args.model_dir), counts_layer=None,
                        ensembl_col="gene_ids", dictionary_dir=str(args.dictionary_dir) if args.dictionary_dir else None,
                        obsm_key=args.obsm_key, model_input_size=args.model_input_size,
                        nproc=args.nproc, forward_batch_size=args.forward_batch_size,
                    )
                    embedding = np.asarray(batch.obsm[args.obsm_key], dtype=np.float32)
                    _validate_embedding(embedding, end - start, dimension)
                    seconds = time.monotonic() - started
                    _checkpoint(checkpoint_path, embedding, identity, seconds)
                    new_batches += 1
                    del batch, counts
                _validate_embedding(embedding, end - start, dimension)
                dimension = int(embedding.shape[1])
                elapsed += seconds
                embedded_cells += end - start
                complete.append((checkpoint_path, identity))
                print(json.dumps({"event": "batch_complete", "context": entry["context"], "batch": batch_index, "n_batches": n_batches, "cells": end - start, "seconds": round(seconds, 3), "estimated_chunk_hours": round(elapsed / embedded_cells * len(cells) / 3600, 3)}), flush=True)
                del embedding
            timing = {"context": entry["context"], "completed_batches": len(complete), "total_batches": n_batches, "embedded_cells": embedded_cells, "total_cells": len(cells), "extraction_seconds": elapsed, "cells_per_second": embedded_cells / elapsed if elapsed else 0, "complete": len(complete) == n_batches}
            _atomic_json(work / "timing.json", timing)
            if len(complete) != n_batches:
                print(json.dumps({"event": "pilot_or_partial", **timing}), flush=True)
                return False

        if _chunk_provenance(args, entry, configuration) != provenance:
            raise ValueError("Source data changed while extraction was running.")
        if _run_configuration(args) != configuration:
            raise ValueError("Model assets or extraction configuration changed while running.")

        # Copying HDF5 keeps all original matrices and annotation encodings intact.
        # Embeddings alone are assembled on disk, so memory remains one batch.
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".h5ad", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(entry["path"], temporary)
            with h5py.File(temporary, "r+") as handle:
                group = handle.require_group("obsm")
                group.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
                if args.obsm_key in group:
                    del group[args.obsm_key]
                dataset = group.create_dataset(args.obsm_key, shape=(int(entry["n_obs"]), dimension), dtype="float32", chunks=(min(args.batch_size, int(entry["n_obs"])), dimension), compression="lzf")
                dataset.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
                for path, identity in complete:
                    values, _ = _load_checkpoint(path, identity)
                    _validate_embedding(values, identity["end"] - identity["start"], dimension)
                    dataset[identity["start"]:identity["end"]] = values
                uns = handle.require_group("uns")
                uns.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
                write_elem(uns, PROVENANCE_KEY, json.dumps(provenance, sort_keys=True))
            _validate_output(temporary, provenance)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        _atomic_json(receipt_path, _validate_output(output, provenance))
        print(json.dumps({"event": "chunk_complete", "context": entry["context"], "output": str(output)}), flush=True)
        return True


def finalize(args: argparse.Namespace, source_manifest: dict[str, Any], entries: list[dict[str, Any]], configuration: dict[str, Any]) -> Path:
    if args.max_batches is not None or args.max_cells is not None:
        raise ValueError("Pilot limits cannot be used with --finalize.")
    final_entries = []
    dimensions = set()
    gene_orders = set()
    for entry in entries:
        _, output, receipt_path = _chunk_paths(args, entry)
        if not output.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Chunk is incomplete: {entry['context']}; no complete manifest was published.")
        provenance = _chunk_provenance(args, entry, configuration)
        receipt = _validate_output(output, provenance, json.loads(receipt_path.read_text()))
        dimensions.add(receipt["embedding_dimension"])
        gene_orders.add(provenance["var_sha256"])
        enriched = copy.deepcopy(entry)
        enriched.update({"source_path": entry["path"], "path": output.relative_to(args.output_dir).as_posix(), "embedding_receipt": receipt_path.relative_to(args.output_dir).as_posix()})
        final_entries.append(enriched)
    if len(dimensions) != 1:
        raise ValueError("Embedding dimensions differ across chunks.")
    if len(gene_orders) != 1:
        raise ValueError("Processed gene identities or order differ across chunks.")
    manifest = copy.deepcopy(source_manifest)
    manifest.pop("contexts", None)
    manifest.update({"chunks": final_entries, "n_chunks": len(final_entries), "n_obs_total": sum(int(entry["n_obs"]) for entry in final_entries), "processed_dir": ".", "chunk_dir": "chunks", "source_processed_manifest": str(args.manifest.resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat(), "geneformer": {"complete": True, "configuration": configuration, "embedding_dimension": dimensions.pop(), "obsm_key": args.obsm_key}})
    path = args.output_dir / "processed_manifest.json"
    _atomic_json(path, manifest)
    print(json.dumps({"event": "manifest_complete", "path": str(path), "n_cells": manifest["n_obs_total"], "n_chunks": len(entries)}), flush=True)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dictionary-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--chunk-index", type=int, help="Zero-based index in source manifest order, suitable for SLURM_ARRAY_TASK_ID.")
    operation.add_argument("--finalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096, help="Cells per resumable extraction batch.")
    parser.add_argument("--forward-batch-size", type=int, default=32, help="Cells per GPU forward pass.")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--model-input-size", type=int, default=4096)
    parser.add_argument("--obsm-key", default="X_geneformer")
    parser.add_argument("--max-batches", type=int, help="Maximum new batches in this invocation; pilot leaves reusable checkpoints.")
    parser.add_argument("--max-cells", type=int, help="Stop before the first complete batch extending past this cell position.")
    args = parser.parse_args(argv)
    for name in ("batch_size", "forward_batch_size", "nproc", "model_input_size", "max_batches", "max_cells"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not args.obsm_key or "/" in args.obsm_key:
        parser.error("--obsm-key must be a nonempty HDF5 key without '/'.")
    for name in ("manifest", "raw_dir", "model_dir", "output_dir", "work_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir == args.manifest.parent:
        parser.error("--output-dir must differ from the source manifest directory.")
    return args


def main() -> None:
    args = parse_args()
    source_manifest, entries = _load_manifest(args.manifest)
    configuration = _run_configuration(args)
    if args.finalize:
        finalize(args, source_manifest, entries, configuration)
    else:
        if not 0 <= args.chunk_index < len(entries):
            raise ValueError(f"--chunk-index must be in [0, {len(entries) - 1}].")
        extract_chunk(args, entries[args.chunk_index], configuration)


if __name__ == "__main__":
    main()
