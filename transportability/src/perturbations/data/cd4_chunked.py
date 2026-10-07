"""Chunked CD4 dataset utilities for metadata-first split planning and subset materialization."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata.experimental import AnnCollection
from scipy import sparse

_IMPLICIT_X_LAYER_KEY = "normalized_log1p"
_TIMEPOINT_ORDER = ("Rest", "Stim8hr", "Stim48hr")
_PLANNING_OBS_COLUMNS = (
    "perturbation",
    "condition",
    "donor",
    "timepoint",
    "donor_timepoint",
    "context",
)
_DERIVED_DONOR_TIMEPOINT_COLUMNS = ("donor", "timepoint")


def _add_donor_timepoint_column(obs: pd.DataFrame) -> None:
    """Add donor-timepoint metadata when older CD4 chunks do not store it."""
    if "donor_timepoint" in obs.columns:
        return

    missing_columns = [
        column for column in _DERIVED_DONOR_TIMEPOINT_COLUMNS if column not in obs.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"CD4 obs is missing columns required for donor_timepoint: {missing}")

    donor = obs["donor"].astype(str)
    timepoint = obs["timepoint"].astype(str)
    obs["donor_timepoint"] = donor.str.cat(timepoint, sep="_")


@dataclass(frozen=True)
class ChunkRecord:
    """One logical CD4 chunk with global row offsets."""

    path: Path
    context: str
    donor: str
    timepoint: str
    chunk_index_within_context: int
    n_obs: int
    n_vars: int
    row_start: int
    row_end: int

    @property
    def chunk_key(self) -> str:
        """Stable chunk identifier used for collection keys and unique cell ids."""
        return f"{self.context}_chunk{self.chunk_index_within_context:03d}"


@dataclass
class BackedCollectionHandle:
    """Context-managed AnnCollection plus backing chunk handles."""

    collection: AnnCollection
    chunks: list[ad.AnnData]

    def close(self) -> None:
        """Close all backing HDF5 handles."""
        for chunk in self.chunks:
            if getattr(chunk, "file", None) is not None:
                chunk.file.close()

    def __enter__(self) -> BackedCollectionHandle:
        """Enter context and return self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context and close backing HDF5 handles."""
        self.close()


def _resolve_manifest_entry_path(manifest_path: Path, raw_path: str) -> Path:
    """Resolve absolute or manifest-relative file paths."""
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _timepoint_sort_key(timepoint: str) -> tuple[int, str]:
    order = {label: idx for idx, label in enumerate(_TIMEPOINT_ORDER)}
    return order.get(timepoint, len(order)), timepoint


def _sorted_chunk_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort manifest-derived chunk rows into the canonical CD4 context order."""
    return sorted(
        rows,
        key=lambda row: (
            str(row["donor"]),
            _timepoint_sort_key(str(row["timepoint"])),
            int(row["chunk_index_within_context"]),
        ),
    )


def parse_cd4_chunk_records(manifest_path: str | Path) -> list[ChunkRecord]:
    """Parse either CD4 manifest shape into normalized chunk records."""
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    raw_rows: list[dict[str, Any]] = []
    if manifest.get("chunks"):
        for entry in manifest["chunks"]:
            raw_rows.append(
                {
                    "path": _resolve_manifest_entry_path(manifest_path, str(entry["path"])),
                    "context": str(entry["context"]),
                    "donor": str(entry["donor"]),
                    "timepoint": str(entry["timepoint"]),
                    "chunk_index_within_context": int(entry["chunk_index_within_context"]),
                    "n_obs": int(entry["n_obs"]),
                    "n_vars": int(entry["n_vars"]),
                }
            )
    elif manifest.get("contexts"):
        for entry in manifest["contexts"]:
            raw_rows.append(
                {
                    "path": _resolve_manifest_entry_path(manifest_path, str(entry["path"])),
                    "context": str(entry["context"]),
                    "donor": str(entry["donor"]),
                    "timepoint": str(entry["timepoint"]),
                    "chunk_index_within_context": 0,
                    "n_obs": int(entry["n_obs"]),
                    "n_vars": int(entry["n_vars"]),
                }
            )
    else:
        raise ValueError(
            f"Unsupported CD4 manifest format: {manifest_path}. "
            "Expected either non-empty 'chunks' or non-empty 'contexts'."
        )

    sorted_rows = _sorted_chunk_rows(raw_rows)
    records: list[ChunkRecord] = []
    row_start = 0
    for row in sorted_rows:
        path = Path(row["path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Chunk file does not exist: {path}")
        n_obs = int(row["n_obs"])
        if n_obs < 0:
            raise ValueError(f"Chunk {path} has negative n_obs={n_obs}.")
        records.append(
            ChunkRecord(
                path=path,
                context=str(row["context"]),
                donor=str(row["donor"]),
                timepoint=str(row["timepoint"]),
                chunk_index_within_context=int(row["chunk_index_within_context"]),
                n_obs=n_obs,
                n_vars=int(row["n_vars"]),
                row_start=row_start,
                row_end=row_start + n_obs,
            )
        )
        row_start += n_obs

    if not records:
        raise ValueError(f"No chunk records found in manifest: {manifest_path}")

    n_vars_set = {record.n_vars for record in records}
    if len(n_vars_set) != 1:
        raise ValueError(f"Chunk manifests disagree on n_vars: {sorted(n_vars_set)}")
    return records


def _copy_matrix(matrix: Any) -> Any:
    """Copy a dense or sparse matrix into standard in-memory form."""
    if sparse.issparse(matrix):
        return matrix.copy().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32).copy()


def _unique_obs_names(index: pd.Index, chunk_key: str) -> pd.Index:
    """Build chunk-stable unique observation names."""
    return pd.Index([f"{name}-{chunk_key}" for name in index.astype(str)], dtype=object)


def _validate_embedding(
    matrix: Any,
    *,
    key: str,
    path: Path,
    n_obs: int,
    expected_width: int | None,
) -> int:
    """Reject malformed embeddings before a split can hide a bad source chunk."""
    shape = getattr(matrix, "shape", ())
    if len(shape) != 2 or shape[0] != n_obs or shape[1] <= 0:
        raise ValueError(
            f"{path}: obsm[{key!r}] must have shape ({n_obs}, positive width); got {shape}."
        )
    width = int(shape[1])
    if expected_width is not None and width != expected_width:
        raise ValueError(f"{path}: obsm[{key!r}] has width {width}; expected {expected_width}.")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{path}: obsm[{key!r}] must contain real numeric values.")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: obsm[{key!r}] contains non-finite values.")
    if values.size and np.abs(values).max() > np.finfo(np.float32).max:
        raise ValueError(f"{path}: obsm[{key!r}] exceeds the float32 range used by STATE.")
    return width


class CD4ChunkedDataset:
    """Chunk-aware CD4 dataset runtime for metadata-first split planning."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        records: list[ChunkRecord],
        obs: pd.DataFrame,
        empty_obs_template: pd.DataFrame,
        var: pd.DataFrame,
        available_layers: tuple[str, ...],
        log1p_uns: Any | None,
        obsm_widths: dict[str, int] | None = None,
    ) -> None:
        """Initialize a chunked CD4 dataset from manifest and metadata."""
        self.manifest_path = manifest_path
        self.records = records
        self.obs = obs
        self._empty_obs_template = empty_obs_template
        self.var = var
        self.available_layers = available_layers
        self._log1p_uns = log1p_uns
        self.obsm_widths = dict(obsm_widths or {})
        self._row_ends = np.asarray([record.row_end for record in records], dtype=np.int64)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        required_obsm_keys: tuple[str, ...] = (),
    ) -> CD4ChunkedDataset:
        """Load metadata and validate requested embeddings across every source chunk."""
        manifest_path = Path(manifest_path).resolve()
        records = parse_cd4_chunk_records(manifest_path)
        planning_obs, obsm_widths = cls._load_planning_obs(records, required_obsm_keys)
        empty_obs_template, var, available_layers, log1p_uns = cls._load_reference_metadata(
            records[0].path
        )
        return cls(
            manifest_path=manifest_path,
            records=records,
            obs=planning_obs,
            empty_obs_template=empty_obs_template,
            var=var,
            available_layers=available_layers,
            log1p_uns=log1p_uns,
            obsm_widths=obsm_widths,
        )

    @staticmethod
    def _load_reference_metadata(
        chunk_path: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], Any | None]:
        """Load reusable empty obs template, var metadata, and layer metadata."""
        adata = ad.read_h5ad(chunk_path, backed="r")
        try:
            empty_obs_template = adata.obs.iloc[0:0].copy()
            _add_donor_timepoint_column(empty_obs_template)
            var = adata.var.copy()
            available_layers = tuple(str(key) for key in adata.layers.keys())
            log1p_uns = copy.deepcopy(adata.uns.get("log1p"))
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        return empty_obs_template, var, available_layers, log1p_uns

    @staticmethod
    def _load_planning_obs(
        records: list[ChunkRecord],
        required_obsm_keys: tuple[str, ...] = (),
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Load only the observation metadata needed for split planning."""
        frames: list[pd.DataFrame] = []
        obsm_widths: dict[str, int] = {}
        for record in records:
            adata = ad.read_h5ad(record.path, backed="r")
            try:
                if adata.n_obs != record.n_obs:
                    raise ValueError(
                        f"{record.path}: manifest n_obs={record.n_obs} does not match "
                        f"the chunk n_obs={adata.n_obs}."
                    )
                for key in dict.fromkeys(required_obsm_keys):
                    if key not in adata.obsm:
                        raise KeyError(f"{record.path}: requested obsm[{key!r}] is missing.")
                    obsm_widths[key] = _validate_embedding(
                        adata.obsm[key],
                        key=key,
                        path=record.path,
                        n_obs=record.n_obs,
                        expected_width=obsm_widths.get(key),
                    )
                obs_columns = set(adata.obs.columns)
                missing_columns = [
                    column
                    for column in _PLANNING_OBS_COLUMNS
                    if column not in obs_columns
                    and not (
                        column == "donor_timepoint"
                        and all(
                            source_column in obs_columns
                            for source_column in _DERIVED_DONOR_TIMEPOINT_COLUMNS
                        )
                    )
                ]
                if missing_columns:
                    missing = ", ".join(missing_columns)
                    raise KeyError(
                        f"{record.path} is missing required planning obs columns: {missing}"
                    )
                available_columns = [
                    column for column in _PLANNING_OBS_COLUMNS if column in adata.obs.columns
                ]
                frame = adata.obs.loc[:, available_columns].copy()
                _add_donor_timepoint_column(frame)
                frame = frame.loc[:, list(_PLANNING_OBS_COLUMNS)]
                frame.index = _unique_obs_names(frame.index, record.chunk_key)
                frames.append(frame)
            finally:
                if getattr(adata, "file", None) is not None:
                    adata.file.close()
        return pd.concat(frames, axis=0, copy=False), obsm_widths

    @property
    def n_obs(self) -> int:
        """Return the total number of observations across all chunks."""
        return int(self.obs.shape[0])

    @property
    def n_vars(self) -> int:
        """Return the total number of variables (genes) across all chunks."""
        return int(self.var.shape[0])

    @property
    def var_names(self) -> pd.Index:
        """Return the variable names (genes) across all chunks."""
        return self.var.index

    def has_source_layer(self, layer_key: str | None) -> bool:
        """Return whether the source chunks can supply the requested layer/X."""
        if layer_key is None:
            return True
        if layer_key in self.available_layers:
            return True
        return layer_key == _IMPLICIT_X_LAYER_KEY

    def open_collection(self) -> BackedCollectionHandle:
        """Open all chunks in backed mode and expose them as one AnnCollection."""
        backed_chunks = [ad.read_h5ad(record.path, backed="r") for record in self.records]
        collection = AnnCollection(
            backed_chunks,
            join_vars="inner",
            label="chunk_id",
            keys=[record.chunk_key for record in self.records],
            index_unique="-",
        )
        return BackedCollectionHandle(collection=collection, chunks=backed_chunks)

    def _resolve_matrix(self, data_obj: Any, layer_key: str | None) -> Any:
        """Resolve the requested matrix from X or a source layer."""
        if layer_key is None:
            return data_obj.X
        if layer_key in data_obj.layers:
            return data_obj.layers[layer_key]
        if layer_key == _IMPLICIT_X_LAYER_KEY:
            return data_obj.X
        available_layers = list(data_obj.layers.keys())
        raise KeyError(
            f"Requested layer '{layer_key}' not found. Available layers: {available_layers}"
        )

    def _materialize_chunk_piece(
        self,
        record: ChunkRecord,
        local_indices: np.ndarray,
        x_layer: str | None,
        include_layers: tuple[str, ...],
    ) -> ad.AnnData:
        """Materialize one chunk-local slice into an ordinary AnnData piece."""
        adata = ad.read_h5ad(record.path, backed="r")
        try:
            # Backed dense HDF5 matrices require increasing, unique row indices.
            sorted_indices, restore_indices = np.unique(local_indices, return_inverse=True)
            view = adata[sorted_indices, :]
            obs = view.obs.copy()
            _add_donor_timepoint_column(obs)
            obs.index = _unique_obs_names(obs.index, record.chunk_key)
            piece = ad.AnnData(
                X=_copy_matrix(self._resolve_matrix(view, x_layer)),
                obs=obs,
                var=self.var.copy(),
            )
            for layer_key in include_layers:
                piece.layers[layer_key] = _copy_matrix(self._resolve_matrix(view, layer_key))
            for key, width in self.obsm_widths.items():
                if key not in view.obsm:
                    raise KeyError(f"{record.path}: requested obsm[{key!r}] is missing.")
                matrix = view.obsm[key]
                _validate_embedding(
                    matrix,
                    key=key,
                    path=record.path,
                    n_obs=piece.n_obs,
                    expected_width=width,
                )
                if sparse.issparse(matrix):
                    matrix = matrix.toarray()
                piece.obsm[key] = np.asarray(matrix, dtype=np.float32).copy()
            if self._log1p_uns is not None:
                piece.uns["log1p"] = copy.deepcopy(self._log1p_uns)
            if not np.array_equal(restore_indices, np.arange(piece.n_obs)):
                piece = piece[restore_indices, :].copy()
            return piece
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def materialize_subset(
        self,
        indices: np.ndarray,
        x_layer: str | None,
        include_layers: tuple[str, ...] | list[str] = (),
        output_path: str | Path | None = None,
    ) -> ad.AnnData:
        """Materialize selected global row indices into an in-memory AnnData subset."""
        row_idx = np.asarray(indices, dtype=np.int64).ravel()
        include_layers_tuple = tuple(dict.fromkeys(str(layer) for layer in include_layers))
        if row_idx.size == 0:
            subset = ad.AnnData(
                X=sparse.csr_matrix((0, self.n_vars), dtype=np.float32),
                obs=self._empty_obs_template.copy(),
                var=self.var.copy(),
            )
            for layer_key in include_layers_tuple:
                subset.layers[layer_key] = sparse.csr_matrix((0, self.n_vars), dtype=np.float32)
            for key, width in self.obsm_widths.items():
                subset.obsm[key] = np.empty((0, width), dtype=np.float32)
            if self._log1p_uns is not None:
                subset.uns["log1p"] = copy.deepcopy(self._log1p_uns)
            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                subset.write_h5ad(output_path)
            return subset

        if row_idx.min() < 0 or row_idx.max() >= self.n_obs:
            raise IndexError(f"Subset indices out of bounds for dataset with n_obs={self.n_obs}.")

        chunk_ids = np.searchsorted(self._row_ends, row_idx, side="right")
        positions_by_chunk: dict[int, list[int]] = defaultdict(list)
        for position, chunk_id in enumerate(chunk_ids.tolist()):
            positions_by_chunk[int(chunk_id)].append(position)

        pieces: list[ad.AnnData] = []
        piece_positions: list[np.ndarray] = []
        for chunk_id in sorted(positions_by_chunk):
            record = self.records[chunk_id]
            positions = np.asarray(positions_by_chunk[chunk_id], dtype=np.int64)
            local_indices = row_idx[positions] - record.row_start
            pieces.append(
                self._materialize_chunk_piece(
                    record=record,
                    local_indices=local_indices,
                    x_layer=x_layer,
                    include_layers=include_layers_tuple,
                )
            )
            piece_positions.append(positions)

        subset = (
            pieces[0]
            if len(pieces) == 1
            else ad.concat(
                pieces,
                axis=0,
                join="inner",
                merge="same",
                uns_merge="same",
                index_unique=None,
            )
        )
        concat_positions = np.concatenate(piece_positions)
        restore_order = np.argsort(concat_positions, kind="stable")
        if not np.array_equal(restore_order, np.arange(restore_order.size, dtype=np.int64)):
            subset = subset[restore_order, :].copy()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subset.write_h5ad(output_path)
        return subset

    def dataset_sparsity(self, x_layer: str | None = None) -> float:
        """Compute sparsity by streaming one chunk at a time."""
        total_entries = int(self.n_obs) * int(self.n_vars)
        if total_entries == 0:
            return float("nan")

        nonzero = 0
        for record in self.records:
            adata = ad.read_h5ad(record.path)
            try:
                matrix = self._resolve_matrix(adata, x_layer)
                if sparse.issparse(matrix):
                    nonzero += int(matrix.nnz)
                else:
                    nonzero += int(np.count_nonzero(np.asarray(matrix)))
            finally:
                del adata
        return 1.0 - (nonzero / total_entries)
