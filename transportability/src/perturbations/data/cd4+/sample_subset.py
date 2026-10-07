"""Create per-context CD4 subset files from processed chunks."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CONTROL_LABEL = "control"
CONDITION_COLUMN = "condition"
PERTURBED_GENE_NAME_COLUMN = "perturbed_gene_name"
DONOR_COLUMN = "donor"
TIMEPOINT_COLUMN = "timepoint"
DONOR_TIMEPOINT_COLUMN = "donor_timepoint"
MANIFEST_NAME = "processed_manifest.json"
TIMEPOINT_ORDER = ("Rest", "Stim8hr", "Stim48hr")


@dataclass(frozen=True)
class ChunkEntry:
    """Metadata for one processed chunk file."""

    path: Path
    context: str
    donor: str
    timepoint: str
    chunk_index_within_context: int


@dataclass(frozen=True)
class PerturbationSelection:
    """Summary for one selected perturbation."""

    condition: str
    perturbed_gene_name: str
    min_count_per_context: int
    total_count: int


@dataclass(frozen=True)
class PerturbationFilter:
    """Resolved perturbation-selection strategy."""

    mode: str
    requested_n_common_perturbations: int | None
    selected_conditions: frozenset[str] | None
    selected_perturbations: tuple[PerturbationSelection, ...]
    selection_rule: str


@dataclass(frozen=True)
class ChunkSelection:
    """Precomputed per-chunk indices for subset materialization."""

    chunk: ChunkEntry
    perturbation_indices: np.ndarray
    control_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample a CD4 subset by optionally keeping the top shared perturbations across "
            "all contexts and sampling a fixed number of control cells per donor/timepoint."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_64",
        help=f"Directory containing {MANIFEST_NAME} and processed chunks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_64_subset",
        help=f"Directory where {MANIFEST_NAME} and chunks/ will be written.",
    )
    parser.add_argument(
        "--n-common-perturbations",
        type=int,
        default=None,
        help="Number of shared non-control perturbations to keep. Omit to keep all perturbations.",
    )
    parser.add_argument(
        "--control-number",
        type=int,
        required=True,
        help="Number of control cells to sample within each donor/timepoint context.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for control-cell sampling.",
    )
    return parser.parse_args()


def load_source_manifest(processed_dir: Path) -> tuple[Path, dict]:
    """
    Load the source processed manifest.

    Args:
        processed_dir: Directory containing the source processed manifest.

    Returns:
        Tuple of the resolved manifest path and the parsed manifest dictionary.
    """
    manifest_path = (processed_dir / MANIFEST_NAME).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing source manifest at {manifest_path}. Run process_data.py first."
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        return manifest_path, json.load(handle)


def resolve_manifest_path(processed_dir: Path, raw_path: str) -> Path:
    """Resolve a path stored in the source manifest."""
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (processed_dir / path).resolve()


def build_chunk_entries(processed_dir: Path, manifest: dict) -> list[ChunkEntry]:
    """Build validated chunk entries from the source manifest."""
    chunks = manifest.get("chunks", [])
    if not chunks:
        raise ValueError("Source manifest contains no chunk entries.")

    entries: list[ChunkEntry] = []
    for chunk in chunks:
        for field in ("path", "context", "donor", "timepoint", "chunk_index_within_context"):
            if field not in chunk:
                raise KeyError(f"Chunk manifest entry is missing required field `{field}`.")

        path = resolve_manifest_path(processed_dir, chunk["path"])
        if not path.exists():
            raise FileNotFoundError(f"Chunk file listed in manifest does not exist: {path}")

        entries.append(
            ChunkEntry(
                path=path,
                context=str(chunk["context"]),
                donor=str(chunk["donor"]),
                timepoint=str(chunk["timepoint"]),
                chunk_index_within_context=int(chunk["chunk_index_within_context"]),
            )
        )

    timepoint_rank = {timepoint: index for index, timepoint in enumerate(TIMEPOINT_ORDER)}
    entries.sort(
        key=lambda entry: (
            entry.donor,
            timepoint_rank.get(entry.timepoint, len(timepoint_rank)),
            entry.chunk_index_within_context,
        )
    )
    return entries


def group_chunks_by_context(chunk_entries: list[ChunkEntry]) -> dict[str, list[ChunkEntry]]:
    """Group chunk entries by donor/timepoint context."""
    grouped: dict[str, list[ChunkEntry]] = defaultdict(list)
    for entry in chunk_entries:
        grouped[entry.context].append(entry)

    return {
        context: sorted(entries, key=lambda entry: entry.chunk_index_within_context)
        for context, entries in grouped.items()
    }


def load_obs_columns(chunk_path: Path, columns: list[str]) -> pd.DataFrame:
    """Load selected observation metadata columns from one chunk."""
    adata = ad.read_h5ad(chunk_path, backed="r")
    try:
        missing_columns = [column for column in columns if column not in adata.obs.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise KeyError(f"{chunk_path} is missing required obs columns: {missing}")
        return adata.obs.loc[:, columns].copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def collect_condition_counts(
    context_chunks: dict[str, list[ChunkEntry]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Count perturbations per context and map condition IDs to gene names."""
    condition_names: dict[str, str] = {}
    counts_by_context: dict[str, dict[str, int]] = {}

    for context, chunks in context_chunks.items():
        context_counts: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            obs = load_obs_columns(chunk.path, [CONDITION_COLUMN, PERTURBED_GENE_NAME_COLUMN])
            chunk_counts = obs[CONDITION_COLUMN].astype(str).value_counts()
            for condition, count in chunk_counts.items():
                context_counts[str(condition)] += int(count)

            name_table = (
                obs.drop_duplicates(subset=CONDITION_COLUMN)
                .set_index(CONDITION_COLUMN)[PERTURBED_GENE_NAME_COLUMN]
                .astype(str)
            )
            for condition, gene_name in name_table.items():
                condition_str = str(condition)
                gene_name_str = str(gene_name)
                previous = condition_names.get(condition_str)
                if previous is not None and previous != gene_name_str:
                    raise ValueError(
                        f"Condition {condition_str} maps to multiple gene names: "
                        f"{previous} vs {gene_name_str}."
                    )
                condition_names[condition_str] = gene_name_str

        counts_by_context[context] = dict(context_counts)

    counts_df = pd.DataFrame(
        {context: counts_by_context[context] for context in context_chunks}
    ).fillna(0)
    return counts_df.astype(np.int64), condition_names


def select_common_perturbations(
    condition_counts: pd.DataFrame,
    condition_names: dict[str, str],
    n_common_perturbations: int,
) -> tuple[PerturbationSelection, ...]:
    """Choose the shared perturbations kept in the subset."""
    if n_common_perturbations <= 0:
        raise ValueError("`--n-common-perturbations` must be positive when provided.")

    candidate_counts = condition_counts.drop(index=CONTROL_LABEL, errors="ignore")
    common_mask = (candidate_counts > 0).all(axis=1)
    common_counts = candidate_counts.loc[common_mask].copy()

    if len(common_counts) < n_common_perturbations:
        raise ValueError(
            f"Requested {n_common_perturbations} shared perturbations, but only "
            f"{len(common_counts)} are present in every context."
        )

    ranking = pd.DataFrame(index=common_counts.index.copy())
    ranking["min_count_per_context"] = common_counts.min(axis=1)
    ranking["total_count"] = common_counts.sum(axis=1)
    ranking["perturbed_gene_name"] = ranking.index.to_series().map(condition_names)
    ranking["condition"] = ranking.index.astype(str)
    ranking = ranking.sort_values(
        by=["min_count_per_context", "total_count", "condition"],
        ascending=[False, False, True],
    )

    return tuple(
        PerturbationSelection(
            condition=str(row["condition"]),
            perturbed_gene_name=str(row["perturbed_gene_name"]),
            min_count_per_context=int(row["min_count_per_context"]),
            total_count=int(row["total_count"]),
        )
        for _, row in ranking.head(n_common_perturbations).iterrows()
    )


def resolve_perturbation_filter(
    context_chunks: dict[str, list[ChunkEntry]],
    n_common_perturbations: int | None,
) -> PerturbationFilter:
    """Resolve which perturbations should be kept."""
    if n_common_perturbations is None:
        return PerturbationFilter(
            mode="all_perturbations",
            requested_n_common_perturbations=None,
            selected_conditions=None,
            selected_perturbations=(),
            selection_rule="Keep all non-control perturbations in every context.",
        )

    condition_counts, condition_names = collect_condition_counts(context_chunks)
    selected_perturbations = select_common_perturbations(
        condition_counts=condition_counts,
        condition_names=condition_names,
        n_common_perturbations=n_common_perturbations,
    )
    return PerturbationFilter(
        mode="top_common_perturbations",
        requested_n_common_perturbations=n_common_perturbations,
        selected_conditions=frozenset(selection.condition for selection in selected_perturbations),
        selected_perturbations=selected_perturbations,
        selection_rule=(
            "Keep the top shared non-control perturbations present in every context, "
            "ranked by highest minimum per-context cell count, then by total cell count, "
            "then by condition identifier."
        ),
    )


def build_context_chunk_selections(
    chunks: list[ChunkEntry],
    perturbation_filter: PerturbationFilter,
) -> tuple[list[ChunkSelection], int, int]:
    """Precompute per-chunk indices for selected perturbations and controls."""
    selections: list[ChunkSelection] = []
    total_controls = 0
    total_perturbation_cells = 0

    selected_conditions = perturbation_filter.selected_conditions
    selected_condition_values = None
    if selected_conditions is not None:
        selected_condition_values = np.array(sorted(selected_conditions), dtype=object)

    for chunk in chunks:
        obs = load_obs_columns(chunk.path, [CONDITION_COLUMN])
        conditions = obs[CONDITION_COLUMN].astype(str).to_numpy()

        if selected_condition_values is None:
            perturbation_mask = conditions != CONTROL_LABEL
        else:
            perturbation_mask = np.isin(conditions, selected_condition_values)

        perturbation_indices = np.flatnonzero(perturbation_mask)
        control_indices = np.flatnonzero(conditions == CONTROL_LABEL)

        selections.append(
            ChunkSelection(
                chunk=chunk,
                perturbation_indices=perturbation_indices,
                control_indices=control_indices,
            )
        )
        total_controls += int(control_indices.size)
        total_perturbation_cells += int(perturbation_indices.size)

    return selections, total_controls, total_perturbation_cells


def sample_control_positions(
    total_controls: int,
    control_number: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample global control positions for one context."""
    if control_number < 0:
        raise ValueError("`--control-number` must be non-negative.")

    if total_controls == 0 or control_number == 0:
        return np.empty(0, dtype=np.int64)

    n_sample = min(total_controls, control_number)
    if n_sample == total_controls:
        return np.arange(total_controls, dtype=np.int64)

    sampled = rng.choice(total_controls, size=n_sample, replace=False)
    return np.sort(sampled.astype(np.int64, copy=False))


def split_sampled_controls_by_chunk(
    chunk_selections: list[ChunkSelection],
    sampled_control_positions: np.ndarray,
) -> list[np.ndarray]:
    """Map sampled global control positions back to chunk-local cell indices."""
    sampled_by_chunk: list[np.ndarray] = []
    offset = 0

    for selection in chunk_selections:
        n_chunk_controls = int(selection.control_indices.size)
        left = np.searchsorted(sampled_control_positions, offset, side="left")
        right = np.searchsorted(sampled_control_positions, offset + n_chunk_controls, side="left")
        local_positions = sampled_control_positions[left:right] - offset
        sampled_by_chunk.append(selection.control_indices[local_positions])
        offset += n_chunk_controls

    return sampled_by_chunk


def materialize_chunk_subset(chunk_path: Path, local_indices: np.ndarray) -> ad.AnnData | None:
    """Materialize a chunk subset into memory."""
    if local_indices.size == 0:
        return None

    adata = ad.read_h5ad(chunk_path, backed="r")
    try:
        return adata[local_indices].to_memory()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def add_donor_timepoint_column(adata: ad.AnnData) -> None:
    """Add a donor-timepoint identifier to observation metadata."""
    missing_columns = [
        column for column in (DONOR_COLUMN, TIMEPOINT_COLUMN) if column not in adata.obs.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Subset is missing required obs columns for donor_timepoint: {missing}")

    donor = adata.obs[DONOR_COLUMN].astype(str)
    timepoint = adata.obs[TIMEPOINT_COLUMN].astype(str)
    adata.obs[DONOR_TIMEPOINT_COLUMN] = donor.str.cat(timepoint, sep="_")


def build_context_subset(
    context: str,
    chunk_selections: list[ChunkSelection],
    sampled_control_positions: np.ndarray,
    perturbation_filter: PerturbationFilter,
    control_number: int,
) -> ad.AnnData:
    """Build the final subset for one donor/timepoint context."""
    sampled_controls_by_chunk = split_sampled_controls_by_chunk(
        chunk_selections, sampled_control_positions
    )

    subset_pieces: list[ad.AnnData] = []
    for selection, sampled_control_indices in zip(
        chunk_selections, sampled_controls_by_chunk, strict=True
    ):
        local_indices = np.concatenate([selection.perturbation_indices, sampled_control_indices])
        if local_indices.size == 0:
            continue

        local_indices.sort()
        piece = materialize_chunk_subset(selection.chunk.path, local_indices)
        if piece is not None:
            subset_pieces.append(piece)

    if not subset_pieces:
        raise ValueError(f"No cells were selected for context {context}.")

    if len(subset_pieces) == 1:
        subset = subset_pieces[0]
    else:
        subset = ad.concat(
            subset_pieces,
            axis=0,
            join="inner",
            merge="same",
            uns_merge="same",
            index_unique=None,
        )

    add_donor_timepoint_column(subset)
    subset.uns["subset_selection"] = {
        "context": context,
        "control_label": CONTROL_LABEL,
        "control_number": int(control_number),
        "filter_mode": perturbation_filter.mode,
        "selection_rule": perturbation_filter.selection_rule,
        "requested_n_common_perturbations": perturbation_filter.requested_n_common_perturbations,
        "selected_conditions": (
            sorted(perturbation_filter.selected_conditions)
            if perturbation_filter.selected_conditions is not None
            else None
        ),
        "selected_perturbed_gene_names": [
            selection.perturbed_gene_name
            for selection in perturbation_filter.selected_perturbations
        ],
    }
    return subset


def build_subset_manifest(
    source_manifest: dict,
    source_manifest_path: Path,
    output_root: Path,
    output_chunk_dir: Path,
    context_records: list[dict[str, object]],
    perturbation_filter: PerturbationFilter,
    control_number: int,
) -> dict:
    """Update the source manifest with subset output information."""
    manifest = copy.deepcopy(source_manifest)
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["processed_dir"] = str(output_root)
    manifest["chunk_dir"] = str(output_chunk_dir)
    manifest["n_chunks"] = len(context_records)
    manifest["n_obs_total"] = int(sum(record["n_obs"] for record in context_records))
    manifest["source_processed_manifest"] = str(source_manifest_path)
    manifest["subset_filter_mode"] = perturbation_filter.mode
    manifest["subset_selection_rule"] = perturbation_filter.selection_rule
    manifest["subset_requested_n_common_perturbations"] = (
        perturbation_filter.requested_n_common_perturbations
    )
    manifest["subset_control_number"] = int(control_number)
    manifest["subset_n_sampled_controls_total"] = int(
        sum(record["n_sampled_controls"] for record in context_records)
    )
    manifest["subset_n_selected_perturbation_cells_total"] = int(
        sum(record["n_selected_perturbation_cells"] for record in context_records)
    )
    manifest["subset_n_selected_perturbations"] = (
        len(perturbation_filter.selected_perturbations)
        if perturbation_filter.selected_conditions is not None
        else source_manifest.get("n_perturbations_after_min_cell_filter")
    )
    manifest["subset_selected_perturbations"] = (
        [
            {
                "condition": selection.condition,
                "perturbed_gene_name": selection.perturbed_gene_name,
                "min_count_per_context": selection.min_count_per_context,
                "total_count": selection.total_count,
            }
            for selection in perturbation_filter.selected_perturbations
        ]
        if perturbation_filter.selected_perturbations
        else None
    )
    manifest["chunks"] = context_records
    return manifest


def write_json(manifest_path: Path, payload: dict) -> None:
    """Write JSON with UTF-8 encoding and a trailing newline."""
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    """Create the CD4 subset and save per-context files."""
    args = parse_args()

    processed_dir = args.processed_dir.resolve()
    output_root = args.output_root.resolve()
    output_chunk_dir = output_root / "chunks"
    output_root.mkdir(parents=True, exist_ok=True)
    output_chunk_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path, source_manifest = load_source_manifest(processed_dir)
    chunk_entries = build_chunk_entries(processed_dir, source_manifest)
    context_chunks = group_chunks_by_context(chunk_entries)
    perturbation_filter = resolve_perturbation_filter(
        context_chunks=context_chunks,
        n_common_perturbations=args.n_common_perturbations,
    )

    print(f"Perturbation selection: {perturbation_filter.selection_rule}", flush=True)
    if perturbation_filter.selected_perturbations:
        for selection in perturbation_filter.selected_perturbations:
            print(
                f"  {selection.perturbed_gene_name} ({selection.condition}) | "
                f"min per context = {selection.min_count_per_context}, "
                f"total = {selection.total_count}",
                flush=True,
            )

    context_records: list[dict[str, object]] = []
    for context_index, (context, chunks) in enumerate(context_chunks.items()):
        donor = chunks[0].donor
        timepoint = chunks[0].timepoint
        rng = np.random.default_rng(args.seed + context_index)

        chunk_selections, total_controls, total_perturbation_cells = build_context_chunk_selections(
            chunks=chunks,
            perturbation_filter=perturbation_filter,
        )
        sampled_control_positions = sample_control_positions(
            total_controls=total_controls,
            control_number=args.control_number,
            rng=rng,
        )

        subset = build_context_subset(
            context=context,
            chunk_selections=chunk_selections,
            sampled_control_positions=sampled_control_positions,
            perturbation_filter=perturbation_filter,
            control_number=args.control_number,
        )
        output_path = (output_chunk_dir / f"cd4_subset_{context}.h5ad").resolve()
        subset.write_h5ad(output_path, compression="gzip")

        n_control_sampled = int((subset.obs[CONDITION_COLUMN].astype(str) == CONTROL_LABEL).sum())
        context_record: dict[str, object] = {
            "path": str(output_path),
            "context": context,
            "donor": donor,
            "timepoint": timepoint,
            "chunk_index_within_context": 0,
            "n_obs": int(subset.n_obs),
            "n_vars": int(subset.n_vars),
            "n_source_controls": int(total_controls),
            "n_sampled_controls": n_control_sampled,
            "n_selected_perturbation_cells": int(total_perturbation_cells),
        }
        if perturbation_filter.selected_perturbations:
            selected_counts = subset.obs[CONDITION_COLUMN].astype(str).value_counts().to_dict()
            context_record["selected_condition_counts"] = {
                selection.condition: int(selected_counts.get(selection.condition, 0))
                for selection in perturbation_filter.selected_perturbations
            }

        print(
            f"Wrote {output_path.name}: {subset.n_obs} cells "
            f"({total_perturbation_cells} perturbation, {n_control_sampled} control)",
            flush=True,
        )

        context_records.append(context_record)
        del subset
        del chunk_selections
        del sampled_control_positions
        gc.collect()

    subset_manifest = build_subset_manifest(
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        output_root=output_root,
        output_chunk_dir=output_chunk_dir,
        context_records=context_records,
        perturbation_filter=perturbation_filter,
        control_number=args.control_number,
    )
    write_json(output_root / MANIFEST_NAME, subset_manifest)


if __name__ == "__main__":
    main()
