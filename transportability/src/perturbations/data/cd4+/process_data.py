"""Preprocess CD4 perturb-seq contexts into QC-filtered chunked AnnData outputs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from ..preprocess import write_manifest_json

FILE_PATTERN = re.compile(
    r"^(?P<label>(?P<donor>D\d+)_(?P<timepoint>Rest|Stim8hr|Stim48hr))\.assigned_guide\.h5ad$"
)
EXPECTED_DONORS = ("D1", "D2", "D3", "D4")
EXPECTED_TIMEPOINTS = ("Rest", "Stim8hr", "Stim48hr")


@dataclass(frozen=True)
class ContextFile:
    """Metadata for one donor/timepoint source file."""

    label: str
    donor: str
    timepoint: str
    path: Path


def iter_batches(n_items: int, batch_size: int):
    """Yield half-open index ranges that partition ``n_items`` by ``batch_size``."""
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def parse_context_files(data_dir: Path) -> list[ContextFile]:
    """
    Parse context files and validate that we have the expected set of donor/timepoint combinations.

    Arguments:
        data_dir: Path to directory containing the *.assigned_guide.h5ad files.

    Returns:
        List of ContextFile objects, sorted by donor and then timepoint.
    """
    contexts: list[ContextFile] = []
    for path in sorted(data_dir.glob("*.assigned_guide.h5ad")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        contexts.append(
            ContextFile(
                label=match.group("label"),
                donor=match.group("donor"),
                timepoint=match.group("timepoint"),
                path=path,
            )
        )

    expected = {f"{donor}_{tp}" for donor in EXPECTED_DONORS for tp in EXPECTED_TIMEPOINTS}
    found = {context.label for context in contexts}
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing:
        raise FileNotFoundError("Missing expected donor/timepoint files: " + ", ".join(missing))
    if extra:
        raise ValueError("Unexpected context labels found: " + ", ".join(extra))

    timepoint_rank = {tp: i for i, tp in enumerate(EXPECTED_TIMEPOINTS)}
    contexts.sort(key=lambda c: (c.donor, timepoint_rank[c.timepoint]))
    return contexts


def compute_single_guide_mask(obs: pd.DataFrame) -> np.ndarray:
    """
    Find cells with single-guide assignments (perturbing a single gene).

    Assumes that cells with single-guide assignments can be identified by either:
        - `guide_group` column with value "targeting single sgRNA", or
        - `guide_id` column with non-missing values that do not contain "multi" (case-insensitive).

    Arguments:
        obs: DataFrame containing cell metadata (adata.obs).

    Returns:
        Boolean array indicating which cells have single-guide assignments.
    """
    if "guide_group" in obs.columns:
        return (obs["guide_group"].astype(str) == "targeting single sgRNA").to_numpy()

    if "guide_id" in obs.columns:
        guide_id = obs["guide_id"]
        guide_str = guide_id.astype(str)
        return (
            ~guide_id.isna() & ~guide_str.str.contains("multi", case=False, na=False)
        ).to_numpy()

    raise KeyError(
        "Could not infer single-guide assignments (expected `guide_group` or `guide_id`)."
    )


def build_gene_reference(
    contexts: list[ContextFile],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Find intersection of gene IDs across all 12 contexts.

    So from here on, all computations are in a common gene space.

    Arguments:
        contexts: List of ContextFile objects for all contexts.

    Returns:
        Tuple of (canonical_gene_ids, base_var, gene_indexers) where:
        - canonical_gene_ids: Array of gene IDs present in all contexts.
        - base_var: DataFrame containing metadata for the canonical genes, taken from the first context.
        - gene_indexers: Dict mapping context label to array of indices that align the context's genes to the canonical gene order.
    """
    local_gene_ids: dict[str, np.ndarray] = {}
    local_var_tables: dict[str, pd.DataFrame] = {}

    for context in contexts:
        adata = ad.read_h5ad(context.path, backed="r")
        try:
            var = adata.var.copy()
            if "gene_ids" in var.columns:
                gene_ids = var["gene_ids"].astype(str).to_numpy()
            else:
                gene_ids = adata.var_names.astype(str).to_numpy()

            var = var.copy()
            var["gene_id"] = gene_ids
            local_gene_ids[context.label] = gene_ids
            local_var_tables[context.label] = var
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    first_label = contexts[0].label
    common_gene_ids = set(local_gene_ids[first_label])
    for context in contexts[1:]:
        common_gene_ids &= set(local_gene_ids[context.label])

    if not common_gene_ids:
        raise ValueError("No overlapping genes were found across contexts.")

    canonical_gene_ids = np.array(
        [gene_id for gene_id in local_gene_ids[first_label] if gene_id in common_gene_ids],
        dtype=object,
    )

    base_var = local_var_tables[first_label].copy()
    base_var = base_var.drop_duplicates(subset="gene_id", keep="first").set_index("gene_id")
    base_var = base_var.loc[canonical_gene_ids].copy()
    base_var.index.name = "gene_id"
    if "gene_ids" not in base_var.columns:
        base_var["gene_ids"] = base_var.index.astype(str)

    gene_indexers: dict[str, np.ndarray] = {}
    for context in contexts:
        local_index = pd.Index(local_gene_ids[context.label])
        indexer = local_index.get_indexer(canonical_gene_ids)
        if (indexer < 0).any():
            raise ValueError(f"Gene alignment failed for {context.label}.")
        gene_indexers[context.label] = indexer.astype(np.int64)

    return canonical_gene_ids, base_var, gene_indexers


def run_cell_qc_and_gene_stats(
    contexts: list[ContextFile],
    gene_indexers: dict[str, np.ndarray],
    n_genes: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Perform cell quality control (QC) and compute gene-level statistics for each context.

    The QC steps are:
    1. Exclude multi-guide cells and cells with very low gene counts (< 100).
    2. Exclude cells with abnormally low or high library sizes using median-absolute-deviation-based outlier detection.

    The gene-level statistics computed are:
    - gene_ncells: number of cells with nonzero counts for each gene across all contexts.
    - gene_totals: total counts for each gene across all contexts.

    Arguments:
        contexts: List of ContextFile objects for all contexts.
        gene_indexers: Dict mapping context label to array of indices that align the context's genes to the canonical gene order.
        n_genes: Total number of canonical genes.
        batch_size: Number of cells to process in each batch when iterating through the data.

    Returns:
        Tuple of (keep_indices, gene_ncells, gene_totals, qc_df) where:
        - keep_indices: Dict mapping context label to array of cell indices that passed QC for that context.
        - gene_ncells: Array of number of cells with nonzero counts for each gene across all contexts.
        - gene_totals: Array of total counts for each gene across all contexts.
        - qc_df: DataFrame summarizing QC metrics for each context.
    """
    keep_indices: dict[str, np.ndarray] = {}
    gene_ncells = np.zeros(n_genes, dtype=np.int64)
    gene_totals = np.zeros(n_genes, dtype=np.float64)
    qc_records: list[dict[str, float | int | str]] = []

    for context in contexts:
        print(f"[Step 2.1] Cell QC: {context.label}")
        adata = ad.read_h5ad(context.path, backed="r")
        print(f"  Total cells in original data: {adata.n_obs}")
        try:
            obs = adata.obs

            # exclude multi-guide cells and cells with very low gene counts (< 100)
            single_guide_mask = compute_single_guide_mask(obs)
            min_genes_mask = obs["n_genes_by_counts"].to_numpy() >= 100
            pre_umi_mask = single_guide_mask & min_genes_mask

            total_counts = obs["total_counts"].to_numpy(dtype=np.float64, copy=False)
            if not np.any(pre_umi_mask):
                raise ValueError(
                    f"{context.label}: no cells after single-guide + min-gene filters."
                )

            # Exclude cells with abnormally low or high library sizes using median-absolute-deviation-based outlier detection
            # keep cells if counts >= max(1400, median_counts - 9 * MAD) and counts <= median_counts + 9 * MAD
            counts_for_bounds = total_counts[pre_umi_mask]
            median_counts = float(np.median(counts_for_bounds))
            mad_counts = float(np.median(np.abs(counts_for_bounds - median_counts)))
            lower_bound = max(1400.0, median_counts - 9.0 * mad_counts)
            upper_bound = median_counts + 9.0 * mad_counts
            umi_mask = (total_counts >= lower_bound) & (total_counts <= upper_bound)

            keep_mask = pre_umi_mask & umi_mask
            keep_idx = np.flatnonzero(keep_mask).astype(np.int64)
            keep_indices[context.label] = keep_idx

            qc_records.append(
                {
                    "context": context.label,
                    "donor": context.donor,
                    "timepoint": context.timepoint,
                    "n_obs_raw": int(adata.n_obs),
                    "n_single_guide": int(single_guide_mask.sum()),
                    "n_min_genes": int(min_genes_mask.sum()),
                    "n_after_cell_qc": int(keep_idx.size),
                    "total_counts_median": median_counts,
                    "total_counts_mad": mad_counts,
                    "total_counts_lower_bound": lower_bound,
                    "total_counts_upper_bound": upper_bound,
                }
            )

            global_to_local = gene_indexers[context.label]
            for start, end in iter_batches(keep_idx.size, batch_size):
                row_idx = keep_idx[start:end]
                x_chunk = adata.X[row_idx, :]
                if sparse.issparse(x_chunk):
                    x_chunk = x_chunk.tocsr()
                    chunk_ncells_local = np.asarray(x_chunk.getnnz(axis=0)).ravel()
                else:
                    x_chunk = np.asarray(x_chunk)
                    chunk_ncells_local = np.asarray((x_chunk > 0).sum(axis=0)).ravel()
                    x_chunk = sparse.csr_matrix(x_chunk)

                chunk_totals_local = (
                    np.asarray(x_chunk.sum(axis=0)).ravel().astype(np.float64, copy=False)
                )
                gene_ncells += chunk_ncells_local[global_to_local].astype(np.int64, copy=False)
                gene_totals += chunk_totals_local[global_to_local]
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    return keep_indices, gene_ncells, gene_totals, pd.DataFrame(qc_records)


def allocate_sample_counts(cell_counts: np.ndarray, max_total: int) -> np.ndarray:
    """
    Perform proportional allocation of sample counts for each context within a timepoint, with a specified maximum total count.

    Arguments:
        cell_counts: Array of cell counts for each context within a timepoint.
        max_total: Maximum total count to allocate across contexts.

    Returns:
        Array of allocated sample counts for each context, proportional to the input cell_counts and summing to at most max_total.
    """
    total_cells = int(cell_counts.sum())
    if total_cells <= max_total:
        return cell_counts.astype(np.int64, copy=True)

    raw = cell_counts / total_cells * max_total
    alloc = np.floor(raw).astype(np.int64)
    remainder = max_total - int(alloc.sum())
    if remainder > 0:
        order = np.argsort(raw - alloc)[::-1]
        alloc[order[:remainder]] += 1
    alloc = np.minimum(alloc, cell_counts)
    return alloc


def select_hvgs_by_timepoint(
    contexts: list[ContextFile],
    keep_indices: dict[str, np.ndarray],
    gene_keep_idx: np.ndarray,
    gene_indexers: dict[str, np.ndarray],
    var_table: pd.DataFrame,
    max_cells_per_timepoint: int | None,
    n_top_genes: int,
    seed: int,
) -> dict[str, set[str]]:
    """
    Select highly variable genes (HVGs) separately for each timepoint (rest, 8hr, 48hr).

    using only the cells that passed QC for that timepoint.
    This accounts for potential differences in gene variability across timepoints.

    Arguments:
        contexts: List of ContextFile objects for all contexts.
        keep_indices: Dict mapping context label to array of cell indices that passed QC for that context
        gene_keep_idx: Array of gene indices that passed QC.
        gene_indexers: Dict mapping context label to array of indices that align the context's genes
            to the canonical gene order.
        var_table: DataFrame containing metadata for the canonical genes.
        max_cells_per_timepoint: Maximum number of cells to use for HVG selection for each
            timepoint. If None or <= 0, use all available cells after QC.
        n_top_genes: Number of top HVGs to select for each timepoint.
        seed: Random seed for reproducibility when subsampling cells for HVG selection.

    Returns:
        Dict mapping timepoint to set of selected HVG gene IDs.
    """
    rng = np.random.default_rng(seed)
    hvg_sets: dict[str, set[str]] = {tp: set() for tp in EXPECTED_TIMEPOINTS}

    var_subset = var_table.iloc[gene_keep_idx].copy()
    var_subset.index = var_subset.index.astype(str)

    for timepoint in EXPECTED_TIMEPOINTS:
        tp_contexts = [context for context in contexts if context.timepoint == timepoint]
        cell_counts = np.array(
            [keep_indices[context.label].size for context in tp_contexts], dtype=np.int64
        )
        total_cells = int(cell_counts.sum())
        if total_cells == 0:
            print(f"[Step 2.3] HVG for {timepoint}: skipped (no cells after QC).")
            continue

        target_cells = (
            total_cells
            if max_cells_per_timepoint is None
            else min(total_cells, max_cells_per_timepoint)
        )
        alloc = allocate_sample_counts(cell_counts, target_cells)
        print(
            f"[Step 2.3] HVG for {timepoint}: using {int(alloc.sum())} cells from {total_cells} QC-passed cells."
        )

        matrices = []
        for context, n_take in zip(tp_contexts, alloc, strict=True):
            if n_take <= 0:
                continue
            rows = keep_indices[context.label]
            if n_take < rows.size:
                sampled_rows = rng.choice(rows, size=int(n_take), replace=False)
                sampled_rows.sort()
            else:
                sampled_rows = rows

            adata = ad.read_h5ad(context.path, backed="r")
            try:
                local_cols = gene_indexers[context.label][gene_keep_idx]
                x = adata.X[sampled_rows, :][:, local_cols]
                if sparse.issparse(x):
                    x = x.tocsr()
                else:
                    x = sparse.csr_matrix(x)
                matrices.append(x)
            finally:
                if getattr(adata, "file", None) is not None:
                    adata.file.close()

        if not matrices:
            continue

        x_tp = sparse.vstack(matrices, format="csr")
        adata_tp = ad.AnnData(X=x_tp, var=var_subset.copy())
        adata_tp.layers["counts"] = adata_tp.X.copy()
        sc.pp.normalize_total(adata_tp, target_sum=1e4)
        sc.pp.log1p(adata_tp)
        sc.pp.highly_variable_genes(
            adata_tp,
            flavor="seurat_v3",
            n_top_genes=min(int(n_top_genes), adata_tp.n_vars),
            layer="counts",
            check_values=False,
            inplace=True,
        )
        hvg_genes = adata_tp.var.index[adata_tp.var["highly_variable"].to_numpy()].tolist()
        hvg_sets[timepoint] = set(hvg_genes)

        del adata_tp
        del matrices

    return hvg_sets


def map_targets_to_global(
    target_gene_series: pd.Series,
    gene_id_to_global: dict[str, int],
    gene_keep_mask: np.ndarray,
) -> np.ndarray:
    """Map perturbed gene ids to kept global-gene indices, using ``-1`` for invalid rows."""
    mapped = target_gene_series.map(gene_id_to_global).fillna(-1).astype(np.int64).to_numpy()
    out = np.full(mapped.shape, -1, dtype=np.int64)
    valid = mapped >= 0
    if np.any(valid):
        mapped_valid = mapped[valid]
        out[valid] = np.where(gene_keep_mask[mapped_valid], mapped_valid, -1)
    return out


def compute_perturbation_metrics(
    contexts: list[ContextFile],
    keep_indices: dict[str, np.ndarray],
    gene_indexers: dict[str, np.ndarray],
    gene_keep_idx: np.ndarray,
    gene_keep_mask: np.ndarray,
    global_to_keep: np.ndarray,
    gene_id_to_global: dict[str, int],
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute per-guide knockdown ratios and per-context control means."""
    records: list[dict[str, str | int | float]] = []
    control_means_by_context: dict[str, np.ndarray] = {}

    for context in contexts:
        print(f"[Step 3.1] Perturbation ratios: {context.label}")
        adata = ad.read_h5ad(context.path, backed="r")
        try:
            keep_idx = keep_indices[context.label]
            obs = adata.obs.iloc[keep_idx].copy()
            guide_type = obs["guide_type"].astype(str).to_numpy()
            guide_id = obs["guide_id"].astype(str).to_numpy()
            perturbed_gene_id = obs["perturbed_gene_id"].astype(str).to_numpy()
            perturbed_gene_name = obs["perturbed_gene_name"].astype(str).to_numpy()
            target_global = map_targets_to_global(
                obs["perturbed_gene_id"],
                gene_id_to_global=gene_id_to_global,
                gene_keep_mask=gene_keep_mask,
            )

            local_cols_keep = gene_indexers[context.label][gene_keep_idx]
            control_mask = guide_type == "non-targeting"
            control_rows = keep_idx[control_mask]
            if control_rows.size == 0:
                raise ValueError(f"{context.label}: no non-targeting controls after cell QC.")

            control_sum = np.zeros(gene_keep_idx.size, dtype=np.float64)
            for start, end in iter_batches(control_rows.size, batch_size):
                rows = control_rows[start:end]
                x = adata.X[rows, :][:, local_cols_keep]
                if sparse.issparse(x):
                    x = x.tocsr()
                else:
                    x = sparse.csr_matrix(x)
                control_sum += np.asarray(x.sum(axis=0)).ravel().astype(np.float64, copy=False)

            control_mean = control_sum / float(control_rows.size)
            control_means_by_context[context.label] = control_mean.astype(np.float32, copy=False)

            targeting_valid = (guide_type == "targeting") & (target_global >= 0)
            if not np.any(targeting_valid):
                continue

            guide_sums: dict[str, float] = defaultdict(float)
            guide_counts: dict[str, int] = defaultdict(int)
            global_to_local = gene_indexers[context.label]

            for start, end in iter_batches(keep_idx.size, batch_size):
                chunk_mask = targeting_valid[start:end]
                if not np.any(chunk_mask):
                    continue
                rows = keep_idx[start:end]
                x_chunk = adata.X[rows, :]
                if sparse.issparse(x_chunk):
                    x_chunk = x_chunk.tocsr()
                else:
                    x_chunk = sparse.csr_matrix(x_chunk)

                row_local = np.flatnonzero(chunk_mask)
                target_global_chunk = target_global[start:end][chunk_mask]
                col_local = global_to_local[target_global_chunk]
                expr = (
                    np.asarray(x_chunk[row_local, col_local]).ravel().astype(np.float64, copy=False)
                )
                guide_chunk = guide_id[start:end][chunk_mask]

                uniq_guides, inv = np.unique(guide_chunk, return_inverse=True)
                sums = np.bincount(inv, weights=expr)
                counts = np.bincount(inv)
                for guide, sum_val, count_val in zip(uniq_guides, sums, counts, strict=True):
                    guide_sums[guide] += float(sum_val)
                    guide_counts[guide] += int(count_val)

            meta = (
                pd.DataFrame(
                    {
                        "guide_id": guide_id[targeting_valid],
                        "perturbed_gene_id": perturbed_gene_id[targeting_valid],
                        "perturbed_gene_name": perturbed_gene_name[targeting_valid],
                        "target_global": target_global[targeting_valid],
                    }
                )
                .drop_duplicates(subset="guide_id", keep="first")
                .set_index("guide_id")
            )

            for guide, n_cells in guide_counts.items():
                if guide not in meta.index:
                    continue
                target_gene_global = int(meta.loc[guide, "target_global"])
                keep_pos = int(global_to_keep[target_gene_global])
                control_mean_target = float(control_mean[keep_pos]) if keep_pos >= 0 else np.nan
                perturbed_mean_target = float(guide_sums[guide] / max(n_cells, 1))
                perturbation_ratio = (
                    perturbed_mean_target / control_mean_target
                    if control_mean_target > 0
                    else np.nan
                )
                records.append(
                    {
                        "context": context.label,
                        "donor": context.donor,
                        "timepoint": context.timepoint,
                        "guide_id": guide,
                        "perturbed_gene_id": str(meta.loc[guide, "perturbed_gene_id"]),
                        "perturbed_gene_name": str(meta.loc[guide, "perturbed_gene_name"]),
                        "n_cells_for_ratio": int(n_cells),
                        "control_mean_target_expr": control_mean_target,
                        "perturbed_mean_target_expr": perturbed_mean_target,
                        "perturbation_ratio": perturbation_ratio,
                    }
                )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    return pd.DataFrame(records), control_means_by_context


def select_perturbations_from_ratios(
    metrics_df: pd.DataFrame,
    timepoints: tuple[str, ...],
    min_donors_per_timepoint: int,
    threshold: float,
    rule: str,
) -> set[str]:
    """Select guides that satisfy donor-level knockdown-ratio rules by timepoint."""
    if metrics_df.empty:
        return set()

    finite = metrics_df[np.isfinite(metrics_df["perturbation_ratio"].to_numpy())]
    if finite.empty:
        return set()

    passing = finite[finite["perturbation_ratio"] < threshold]
    donor_counts = passing.groupby(["guide_id", "timepoint"])["donor"].nunique()
    all_guides = sorted(finite["guide_id"].unique())

    kept = set()
    for guide in all_guides:
        counts = [int(donor_counts.get((guide, timepoint), 0)) for timepoint in timepoints]
        if rule == "all":
            should_keep = all(count >= min_donors_per_timepoint for count in counts)
        elif rule == "any":
            should_keep = any(count >= min_donors_per_timepoint for count in counts)
        else:
            raise ValueError(f"Unknown perturbation rule: {rule}")
        if should_keep:
            kept.add(guide)
    return kept


def apply_cell_level_and_min_count_filters(
    contexts: list[ContextFile],
    keep_indices: dict[str, np.ndarray],
    gene_indexers: dict[str, np.ndarray],
    gene_keep_mask: np.ndarray,
    global_to_keep: np.ndarray,
    gene_id_to_global: dict[str, int],
    control_means_by_context: dict[str, np.ndarray],
    kept_perturbations: set[str],
    ratio_threshold: float,
    min_cells_per_perturbation: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, set[str]]:
    """Apply per-cell knockdown filtering and minimum-cell perturbation retention."""
    kept_perturbations_arr = np.array(sorted(kept_perturbations), dtype=object)
    initial_keep_masks: dict[str, np.ndarray] = {}
    per_context_counts: dict[str, dict[str, int]] = {}

    for context in contexts:
        print(f"[Step 3.2] Cell-level knockdown filter: {context.label}")
        adata = ad.read_h5ad(context.path, backed="r")
        try:
            keep_idx = keep_indices[context.label]
            obs = adata.obs.iloc[keep_idx].copy()
            guide_type = obs["guide_type"].astype(str).to_numpy()
            guide_id = obs["guide_id"].astype(str).to_numpy()
            target_global = map_targets_to_global(
                obs["perturbed_gene_id"],
                gene_id_to_global=gene_id_to_global,
                gene_keep_mask=gene_keep_mask,
            )

            control_mask = guide_type == "non-targeting"
            keep_mask = control_mask.copy()

            if kept_perturbations_arr.size > 0:
                candidate_mask = (
                    (guide_type == "targeting")
                    & (target_global >= 0)
                    & np.isin(guide_id, kept_perturbations_arr)
                )
            else:
                candidate_mask = np.zeros(keep_idx.size, dtype=bool)

            global_to_local = gene_indexers[context.label]
            control_mean = control_means_by_context[context.label].astype(np.float64, copy=False)

            for start, end in iter_batches(keep_idx.size, batch_size):
                chunk_mask = candidate_mask[start:end]
                if not np.any(chunk_mask):
                    continue

                rows = keep_idx[start:end]
                x_chunk = adata.X[rows, :]
                if sparse.issparse(x_chunk):
                    x_chunk = x_chunk.tocsr()
                else:
                    x_chunk = sparse.csr_matrix(x_chunk)

                row_local = np.flatnonzero(chunk_mask)
                target_global_chunk = target_global[start:end][chunk_mask]
                col_local = global_to_local[target_global_chunk]
                expr = (
                    np.asarray(x_chunk[row_local, col_local]).ravel().astype(np.float64, copy=False)
                )

                keep_pos = global_to_keep[target_global_chunk]
                denom = control_mean[keep_pos]
                ratios = np.full(expr.shape, np.inf, dtype=np.float64)
                valid = denom > 0
                ratios[valid] = expr[valid] / denom[valid]
                passed_local = row_local[ratios < ratio_threshold]
                keep_mask[start + passed_local] = True

            kept_targeting_guides = guide_id[keep_mask & (guide_type == "targeting")]
            if kept_targeting_guides.size == 0:
                per_context_counts[context.label] = {}
            else:
                counts = pd.Series(kept_targeting_guides).value_counts().astype(int).to_dict()
                per_context_counts[context.label] = counts
            initial_keep_masks[context.label] = keep_mask
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    kept_after_min_cells: set[str] = set()
    for guide in kept_perturbations:
        min_context_cells = min(
            per_context_counts[context.label].get(guide, 0) for context in contexts
        )
        if min_context_cells >= min_cells_per_perturbation:
            kept_after_min_cells.add(guide)

    kept_after_min_cells_arr = np.array(sorted(kept_after_min_cells), dtype=object)
    final_indices: dict[str, np.ndarray] = {}
    summary_records: list[dict[str, str | int]] = []

    for context in contexts:
        keep_idx = keep_indices[context.label]
        initial_keep = initial_keep_masks[context.label]

        adata = ad.read_h5ad(context.path, backed="r")
        try:
            obs = adata.obs.iloc[keep_idx].copy()
            guide_type = obs["guide_type"].astype(str).to_numpy()
            guide_id = obs["guide_id"].astype(str).to_numpy()

            final_keep = initial_keep.copy()
            targeting_kept = (guide_type == "targeting") & final_keep
            if kept_after_min_cells_arr.size > 0:
                drop_mask = targeting_kept & (~np.isin(guide_id, kept_after_min_cells_arr))
            else:
                drop_mask = targeting_kept
            final_keep[drop_mask] = False

            final_idx = keep_idx[final_keep].astype(np.int64)
            final_indices[context.label] = final_idx

            summary_records.append(
                {
                    "context": context.label,
                    "donor": context.donor,
                    "timepoint": context.timepoint,
                    "n_after_cell_qc": int(keep_idx.size),
                    "n_after_cell_knockdown_filter": int(initial_keep.sum()),
                    "n_after_min_cells_filter": int(final_keep.sum()),
                    "n_controls_final": int(np.sum(final_keep & (guide_type == "non-targeting"))),
                    "n_targeting_final": int(np.sum(final_keep & (guide_type == "targeting"))),
                }
            )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    return final_indices, pd.DataFrame(summary_records), kept_after_min_cells


def write_processed_chunks(
    contexts: list[ContextFile],
    final_indices: dict[str, np.ndarray],
    final_gene_idx: np.ndarray,
    gene_indexers: dict[str, np.ndarray],
    var_table: pd.DataFrame,
    output_chunk_dir: Path,
    output_chunk_size: int,
    normalize_target_sum: float,
) -> list[dict[str, str | int]]:
    """Write filtered context matrices to chunked processed AnnData files."""
    chunk_records: list[dict[str, str | int]] = []
    var_final = var_table.iloc[final_gene_idx].copy()
    var_final.index = var_final.index.astype(str)

    for context in contexts:
        print(f"[Step 4] Writing chunks: {context.label}")
        adata = ad.read_h5ad(context.path, backed="r")
        try:
            final_rows = final_indices[context.label]
            local_cols = gene_indexers[context.label][final_gene_idx]

            for chunk_i, (start, end) in enumerate(
                iter_batches(final_rows.size, output_chunk_size)
            ):
                rows = final_rows[start:end]
                x = adata.X[rows, :][:, local_cols]
                if sparse.issparse(x):
                    x = x.tocsr()
                else:
                    x = sparse.csr_matrix(x)

                obs_chunk = adata.obs.iloc[rows].copy()
                guide_type = obs_chunk["guide_type"].astype(str).to_numpy()
                # Export perturbation targets in the same gene-ID namespace as `var_names`.
                perturbation = obs_chunk["perturbed_gene_id"].astype(str).to_numpy()
                perturbation = np.where(guide_type == "non-targeting", "control", perturbation)
                obs_chunk["perturbation"] = pd.Categorical(perturbation)
                obs_chunk["condition"] = obs_chunk["perturbation"].copy()
                obs_chunk["donor"] = context.donor
                obs_chunk["timepoint"] = context.timepoint
                obs_chunk["context"] = context.label

                chunk_adata = ad.AnnData(X=x, obs=obs_chunk, var=var_final.copy())
                chunk_adata.layers["counts"] = chunk_adata.X.copy()
                sc.pp.normalize_total(chunk_adata, target_sum=normalize_target_sum)
                sc.pp.log1p(chunk_adata)
                chunk_adata.uns["source_context"] = context.label
                chunk_adata.uns["source_file"] = context.path.name

                chunk_path = (
                    output_chunk_dir / f"cd4_processed_{context.label}_chunk{chunk_i:03d}.h5ad"
                )
                chunk_adata.write_h5ad(chunk_path, compression="gzip")

                chunk_records.append(
                    {
                        "path": str(chunk_path.resolve()),
                        "context": context.label,
                        "donor": context.donor,
                        "timepoint": context.timepoint,
                        "chunk_index_within_context": int(chunk_i),
                        "n_obs": int(chunk_adata.n_obs),
                        "n_vars": int(chunk_adata.n_vars),
                    }
                )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    return chunk_records


def main():
    """Run end-to-end CD4 preprocessing and write outputs plus summary metadata."""
    parser = argparse.ArgumentParser(
        description="Preprocess CD4+ perturb-seq data in backed/chunked mode."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing raw *.assigned_guide.h5ad files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Directory where processed outputs are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Row batch size for streaming computations.",
    )
    parser.add_argument(
        "--hvg-max-cells-per-timepoint",
        type=int,
        default=200_000,
        help="Max QC-passed cells sampled per timepoint for HVG selection; use 0 to disable sampling cap.",
    )
    parser.add_argument(
        "--hvg-top-genes",
        type=int,
        default=2_000,
        help="Number of HVGs to select per timepoint before union.",
    )
    parser.add_argument(
        "--knockdown-threshold",
        type=float,
        default=0.5,
        help="Threshold for perturbation and cell knockdown ratios.",
    )
    parser.add_argument(
        "--min-donors-per-timepoint",
        type=int,
        default=2,
        help="Minimum donor count with perturbation ratio below threshold.",
    )
    parser.add_argument(
        "--perturbation-timepoint-rule",
        choices=("all", "any"),
        default="all",
        help="`all`: pass threshold in >= min-donors for every timepoint. `any`: for at least one timepoint.",
    )
    parser.add_argument(
        "--min-cells-per-perturbation",
        type=int,
        default=256,
        help="Keep perturbations with at least this many post-filter cells in every donor-timepoint context.",
    )
    parser.add_argument(
        "--output-chunk-size",
        type=int,
        default=250_000,
        help="Maximum cells per saved processed chunk file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for HVG sampling.",
    )
    parser.add_argument(
        "--normalize-target-sum",
        type=float,
        default=1e4,
        help="Target sum for per-cell library-size normalization in saved chunks.",
    )
    args = parser.parse_args()

    print("Starting processing the CD4+ perturb-seq dataset -- in total of 4 steps")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_chunk_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_chunk_dir.mkdir(parents=True, exist_ok=True)

    contexts = parse_context_files(data_dir)
    print(f"Found {len(contexts)} raw context files.")

    canonical_gene_ids, var_table, gene_indexers = build_gene_reference(contexts)
    print(f"[Init] Canonical gene set size: {len(canonical_gene_ids)}")

    keep_indices, gene_ncells, gene_totals, cell_qc_df = run_cell_qc_and_gene_stats(
        contexts=contexts,
        gene_indexers=gene_indexers,
        n_genes=len(canonical_gene_ids),
        batch_size=args.batch_size,
    )

    gene_keep_mask = (gene_ncells >= 100) & (gene_totals >= 100)
    gene_keep_idx = np.flatnonzero(gene_keep_mask)
    if gene_keep_idx.size == 0:
        raise ValueError("Gene QC removed all genes.")
    print(f"[Step 2.2] Genes passing global QC: {gene_keep_idx.size}/{len(canonical_gene_ids)}")
    n_cells_qc = sum(idx.size for idx in keep_indices.values())
    n_cells_raw = cell_qc_df["n_obs_raw"].sum()
    print(f"[Step 2.2] Number of cells passing QC: {n_cells_qc}/{n_cells_raw}")

    hvg_cap = None if args.hvg_max_cells_per_timepoint <= 0 else args.hvg_max_cells_per_timepoint
    hvg_sets = select_hvgs_by_timepoint(
        contexts=contexts,
        keep_indices=keep_indices,
        gene_keep_idx=gene_keep_idx,
        gene_indexers=gene_indexers,
        var_table=var_table,
        max_cells_per_timepoint=hvg_cap,
        n_top_genes=args.hvg_top_genes,
        seed=args.seed,
    )

    canonical_index = pd.Index(canonical_gene_ids)
    hvg_union = set().union(*hvg_sets.values())
    final_gene_idx = canonical_index.get_indexer(pd.Index(list(hvg_union)))
    final_gene_idx = final_gene_idx[final_gene_idx >= 0]
    final_gene_idx = np.sort(np.unique(final_gene_idx))
    if final_gene_idx.size == 0:
        raise ValueError("HVG union is empty. Check preprocessing settings.")
    print(f"[Step 2.3] HVG union size: {final_gene_idx.size}")
    final_gene_ids = set(canonical_gene_ids[final_gene_idx].astype(str).tolist())

    var_table["n_cells_after_cell_qc"] = gene_ncells
    var_table["total_counts_after_cell_qc"] = gene_totals
    var_table["passes_gene_qc"] = gene_keep_mask
    for timepoint in EXPECTED_TIMEPOINTS:
        col_name = f"hvg_{timepoint.lower()}"
        var_table[col_name] = var_table.index.astype(str).isin(hvg_sets.get(timepoint, set()))
    var_table["highly_variable"] = var_table.index.astype(str).isin(hvg_union)

    gene_id_to_global = {gene_id: i for i, gene_id in enumerate(canonical_gene_ids.astype(str))}
    global_to_keep = np.full(len(canonical_gene_ids), -1, dtype=np.int64)
    global_to_keep[gene_keep_idx] = np.arange(gene_keep_idx.size, dtype=np.int64)

    metrics_df, control_means_by_context = compute_perturbation_metrics(
        contexts=contexts,
        keep_indices=keep_indices,
        gene_indexers=gene_indexers,
        gene_keep_idx=gene_keep_idx,
        gene_keep_mask=gene_keep_mask,
        global_to_keep=global_to_keep,
        gene_id_to_global=gene_id_to_global,
        batch_size=args.batch_size,
    )
    metrics_df = metrics_df.loc[
        metrics_df["perturbed_gene_id"].astype(str).isin(final_gene_ids)
    ].copy()
    kept_perturbations = select_perturbations_from_ratios(
        metrics_df=metrics_df,
        timepoints=EXPECTED_TIMEPOINTS,
        min_donors_per_timepoint=args.min_donors_per_timepoint,
        threshold=args.knockdown_threshold,
        rule=args.perturbation_timepoint_rule,
    )
    print(f"[Step 3.1] Perturbations passing donor/timepoint criterion: {len(kept_perturbations)}")

    final_indices, filter_summary_df, kept_after_min_cells = apply_cell_level_and_min_count_filters(
        contexts=contexts,
        keep_indices=keep_indices,
        gene_indexers=gene_indexers,
        gene_keep_mask=gene_keep_mask,
        global_to_keep=global_to_keep,
        gene_id_to_global=gene_id_to_global,
        control_means_by_context=control_means_by_context,
        kept_perturbations=kept_perturbations,
        ratio_threshold=args.knockdown_threshold,
        min_cells_per_perturbation=args.min_cells_per_perturbation,
        batch_size=args.batch_size,
    )
    print(f"[Step 3.3] Perturbations passing min-cell filter: {len(kept_after_min_cells)}")

    chunk_records = write_processed_chunks(
        contexts=contexts,
        final_indices=final_indices,
        final_gene_idx=final_gene_idx,
        gene_indexers=gene_indexers,
        var_table=var_table,
        output_chunk_dir=output_chunk_dir,
        output_chunk_size=args.output_chunk_size,
        normalize_target_sum=args.normalize_target_sum,
    )

    genes_csv_path = output_dir / "cd4_genes.csv.gz"
    var_table.to_csv(genes_csv_path, compression="gzip")

    cell_qc_path = output_dir / "cell_qc_summary.csv"
    cell_qc_df.to_csv(cell_qc_path, index=False)

    filter_summary_path = output_dir / "filter_summary.csv"
    filter_summary_df.to_csv(filter_summary_path, index=False)

    metrics_path = output_dir / "perturbation_knockdown_metrics.csv.gz"
    metrics_df.to_csv(metrics_path, index=False, compression="gzip")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_dir": str(data_dir),
        "processed_dir": str(output_dir),
        "chunk_dir": str(output_chunk_dir),
        "n_raw_contexts": len(contexts),
        "n_chunks": len(chunk_records),
        "n_obs_total": int(sum(record["n_obs"] for record in chunk_records)),
        "n_vars_final": int(final_gene_idx.size),
        "n_genes_pass_gene_qc": int(gene_keep_idx.size),
        "n_perturbations_after_ratio_filter": len(kept_perturbations),
        "n_perturbations_after_min_cell_filter": len(kept_after_min_cells),
        "knockdown_threshold": float(args.knockdown_threshold),
        "min_donors_per_timepoint": int(args.min_donors_per_timepoint),
        "perturbation_timepoint_rule": args.perturbation_timepoint_rule,
        "min_cells_per_perturbation": int(args.min_cells_per_perturbation),
        "hvg_top_genes_per_timepoint": int(args.hvg_top_genes),
        "hvg_max_cells_per_timepoint": None if hvg_cap is None else int(hvg_cap),
        "genes_csv": str(genes_csv_path.resolve()),
        "cell_qc_summary_csv": str(cell_qc_path.resolve()),
        "filter_summary_csv": str(filter_summary_path.resolve()),
        "perturbation_knockdown_metrics_csv": str(metrics_path.resolve()),
        "chunks": chunk_records,
    }
    manifest_path = output_dir / "processed_manifest.json"
    write_manifest_json(manifest_path, manifest)

    print("[Done] Wrote processed outputs:")
    print(f"  - {manifest_path}")
    print(f"  - {genes_csv_path}")
    print(f"  - {cell_qc_path}")
    print(f"  - {filter_summary_path}")
    print(f"  - {metrics_path}")
    print(f"  - chunks/: {len(chunk_records)} files")


if __name__ == "__main__":
    main()
