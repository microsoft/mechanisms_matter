"""
scDesign2-style summary statistics for validating simulated against real data.

These diagnostics quantify how closely a simulated count matrix (e.g. from
CausalDGP) reproduces the marginal and pairwise structure of a real Perturb-seq
dataset. They follow the panel proposed by scDesign2 (Sun et al., Genome Biology
2021):

- **Gene-wise** (per gene, across cells): mean, variance, zero proportion,
  coefficient of variation, and Fano factor (over-dispersion).
- **Cell-wise** (per cell, across genes): zero proportion and library size.
- **Gene-pair** (across a gene subset): Pearson correlation and Kendall's tau of
  the pairwise co-expression structure.

Each function computes statistics for a single dataset.

All statistics are computed on raw counts by default. Pass ``layer`` to use a
specific layer (e.g. ``"counts"`` for a processed AnnData that keeps raw counts
in a layer).
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.stats import kendalltau

from perturbations.util.anndata_util import get_matrix

_EPS = 1e-12


def _as_csr(matrix: Any) -> sparse.csr_matrix | np.ndarray:
    """Return a CSR matrix for sparse input or a 2D float64 array otherwise."""
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float64, copy=False)
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}.")
    return arr


def _is_all_integer(matrix: Any, atol: float = 1e-8) -> bool:
    """
    Return whether all matrix values are (near-)integers, sparse-aware.

    For sparse matrices only the stored nonzero values are checked, since the
    implicit zeros are already integers.
    """
    if sparse.issparse(matrix):
        values = matrix.data
    else:
        values = np.asarray(matrix, dtype=np.float64).ravel()
    if values.size == 0:
        return True
    return bool(np.allclose(values, np.round(values), atol=atol))


def _resolve_counts_matrix(adata: AnnData, layer: str | None) -> sparse.csr_matrix | np.ndarray:
    """
    Return an integer count matrix, falling back to ``layers['counts']``.

    Args:
        adata: AnnData to read from.
        layer: Preferred layer; ``None`` uses ``.X``.

    Returns:
        The selected matrix if it is integer-valued, otherwise the ``counts``
        layer when available.

    Raises:
        ValueError: If neither the selected matrix nor a ``counts`` layer holds
            integer counts.
    """
    matrix = _as_csr(get_matrix(adata, layer))
    if _is_all_integer(matrix):
        return matrix

    layers = getattr(adata, "layers", None)
    if layers is not None and "counts" in layers:
        counts = _as_csr(get_matrix(adata, "counts"))
        if not _is_all_integer(counts):
            raise ValueError("'counts' layer is not integer-valued; expected raw counts.")
        return counts

    raise ValueError(
        "Count statistics require integer counts, but the selected matrix is non-integer "
        "and no 'counts' layer exists. Provide raw counts in .X or layers['counts']."
    )


def _column_moments(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column (per-gene) mean and variance for sparse or dense input."""
    n_rows = matrix.shape[0]
    if n_rows == 0:
        width = matrix.shape[1]
        return np.zeros(width), np.zeros(width)

    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        var = np.maximum(mean_sq - mean**2, 0.0)
        return mean, var

    mean = matrix.mean(axis=0)
    var = matrix.var(axis=0)
    return np.asarray(mean).ravel(), np.asarray(var).ravel()


def gene_wise_statistics(
    adata: AnnData, layer: str | None = None, use_counts: bool = True
) -> pd.DataFrame:
    """
    Compute per-gene mean, variance, zero proportion and coefficient of variation.

    Args:
        adata: AnnData whose matrix (``.X`` or ``layer``) holds counts.
        layer: Optional layer name; ``None`` uses ``.X``.
        use_counts: When ``True``, verify the data is integer counts, falling
            back to ``layers['counts']`` when the selected matrix is not.

    Returns:
        DataFrame indexed by gene (``var_names``) with columns
        ``["mean", "variance", "zero_proportion", "cov", "fano"]``.
    """
    matrix = (
        _resolve_counts_matrix(adata, layer) if use_counts else _as_csr(get_matrix(adata, layer))
    )
    n_cells = int(matrix.shape[0])

    mean, variance = _column_moments(matrix)

    if sparse.issparse(matrix):
        nonzero_per_gene = matrix.getnnz(axis=0).astype(np.float64)
    else:
        nonzero_per_gene = np.count_nonzero(matrix, axis=0).astype(np.float64)
    zero_proportion = (
        1.0 - nonzero_per_gene / n_cells if n_cells > 0 else np.zeros_like(nonzero_per_gene)
    )

    cov = np.sqrt(variance) / (mean + _EPS)
    # Fano factor (variance / mean): the direct over-dispersion measure. It equals
    # 1 under Poisson sampling and exceeds 1 for over-dispersed counts.
    fano = variance / (mean + _EPS)

    return pd.DataFrame(
        {
            "mean": mean,
            "variance": variance,
            "zero_proportion": zero_proportion,
            "cov": cov,
            "fano": fano,
        },
        index=pd.Index(np.asarray(adata.var_names, dtype=str), name="gene"),
    )


def cell_wise_statistics(
    adata: AnnData, layer: str | None = None, use_counts: bool = True
) -> pd.DataFrame:
    """
    Compute per-cell zero proportion and library size.

    Args:
        adata: AnnData whose matrix (``.X`` or ``layer``) holds counts.
        layer: Optional layer name; ``None`` uses ``.X``.
        use_counts: When ``True``, verify the data is integer counts, falling
            back to ``layers['counts']`` when the selected matrix is not.

    Returns:
        DataFrame indexed by cell (``obs_names``) with columns
        ``["zero_proportion", "library_size"]``.
    """
    matrix = (
        _resolve_counts_matrix(adata, layer) if use_counts else _as_csr(get_matrix(adata, layer))
    )
    n_genes = int(matrix.shape[1])

    if sparse.issparse(matrix):
        library_size = np.asarray(matrix.sum(axis=1)).ravel()
        nonzero_per_cell = matrix.getnnz(axis=1).astype(np.float64)
    else:
        library_size = matrix.sum(axis=1)
        nonzero_per_cell = np.count_nonzero(matrix, axis=1).astype(np.float64)
    zero_proportion = (
        1.0 - nonzero_per_cell / n_genes if n_genes > 0 else np.zeros_like(nonzero_per_cell)
    )

    return pd.DataFrame(
        {
            "zero_proportion": zero_proportion,
            "library_size": np.asarray(library_size, dtype=np.float64).ravel(),
        },
        index=pd.Index(np.asarray(adata.obs_names, dtype=str), name="cell"),
    )


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the flattened strict upper triangle of a square matrix."""
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def _select_gene_indices(
    adata: AnnData, layer: str | None, max_genes: int, selection: str
) -> np.ndarray:
    """
    Select gene column indices for the gene-pair block.

    ``"mean"`` ranks genes by mean expression on ``layer``. ``"hvg"`` ranks by
    Seurat v3 highly-variable-gene scores computed on integer counts (falling
    back to ``layers['counts']`` when needed).

    Args:
        adata: AnnData to select genes from.
        layer: Layer used for mean-expression ranking; ``None`` uses ``.X``.
        max_genes: Number of genes to select.
        selection: Either ``"mean"`` or ``"hvg"``.

    Returns:
        Ascending array of selected gene column indices.
    """
    k = min(max_genes, int(adata.n_vars))

    if selection == "mean":
        mean_expr = _column_moments(_as_csr(get_matrix(adata, layer)))[0]
        order = np.argsort(mean_expr)[::-1]
        return np.sort(order[:k])

    if selection == "hvg":
        counts = _resolve_counts_matrix(adata, None)
        tmp = AnnData(counts)
        sc.pp.highly_variable_genes(tmp, n_top_genes=k, flavor="seurat_v3")
        ranked = np.asarray(tmp.var["highly_variable_rank"], dtype=np.float64)
        hv_idx = np.where(~np.isnan(ranked))[0]
        hv_idx = hv_idx[np.argsort(ranked[hv_idx])]
        return np.sort(hv_idx[:k])

    raise ValueError(f"Unknown selection={selection!r}; expected 'mean' or 'hvg'.")


def _assert_log_normalized(matrix: np.ndarray, max_expected: float = 50.0) -> None:
    """
    Assert that ``matrix`` looks log-normalized rather than raw counts.

    Log1p-normalized expression is non-negative, fractional (raw counts are
    integers), and bounded to small values. This guards against the common
    mistake of computing co-expression correlations on raw counts, where shared
    library size spuriously inflates every pairwise correlation.

    Args:
        matrix: Dense expression values to check.
        max_expected: Upper bound above which values are treated as raw counts.
    """
    finite = np.asarray(matrix, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    assert finite.min() >= 0.0, "Expected non-negative log-normalized values, found negatives."
    assert not _is_all_integer(matrix), (
        "Layer looks like integer counts, not log-normalized values. "
        "Pass a log-normalized layer (e.g. layer='normalized_log1p')."
    )
    assert finite.max() <= max_expected, (
        f"Max value {finite.max():.2f} exceeds {max_expected}; the layer may hold raw counts "
        "rather than log-normalized expression."
    )


def _build_metacells(
    full_counts: sparse.csr_matrix | np.ndarray,
    selected_idx: np.ndarray,
    metacell_size: int,
    max_metacells: int,
    rng: np.random.Generator,
    target_sum: float = 1e4,
) -> np.ndarray:
    """
    Aggregate cells into log-normalized metacell profiles for selected genes.

    Cells are randomly partitioned into groups of ``metacell_size``; counts are
    summed per group and normalized by the metacell's total library size (over
    all genes) then log1p-transformed. This denoises single-cell measurement
    noise before computing co-expression correlations.

    Args:
        full_counts: Raw counts for all genes, shape ``(n_cells, n_all_genes)``.
        selected_idx: Column indices of the genes to keep.
        metacell_size: Cells per metacell.
        max_metacells: Cap on the number of metacells used.
        rng: RNG for the partitioning.
        target_sum: Per-metacell total used for normalization.

    Returns:
        Log-normalized ``(n_metacells, len(selected_idx))`` metacell matrix.
    """
    n_cells = int(full_counts.shape[0])
    n_meta = n_cells // metacell_size
    if n_meta < 2:
        raise ValueError(
            f"Not enough cells ({n_cells}) for metacells of size {metacell_size} (need >= 2)."
        )

    if sparse.issparse(full_counts):
        lib = np.asarray(full_counts.sum(axis=1)).ravel().astype(np.float64)
        selected = full_counts[:, selected_idx].tocsr()
    else:
        dense_counts = np.asarray(full_counts, dtype=np.float64)
        lib = dense_counts.sum(axis=1)
        selected = dense_counts[:, selected_idx]

    perm = rng.permutation(n_cells)[: n_meta * metacell_size]
    groups = perm.reshape(n_meta, metacell_size)
    if n_meta > max_metacells:
        keep = rng.choice(n_meta, size=max_metacells, replace=False)
        groups = groups[keep]
        n_meta = groups.shape[0]

    out = np.empty((n_meta, selected_idx.size), dtype=np.float64)
    for m in range(n_meta):
        idx = groups[m]
        if sparse.issparse(selected):
            gene_sum = np.asarray(selected[idx].sum(axis=0)).ravel()
        else:
            gene_sum = selected[idx].sum(axis=0)
        total = float(lib[idx].sum())
        out[m] = np.log1p(gene_sum / total * target_sum) if total > 0 else 0.0
    return out


def gene_pair_correlations(
    adata: AnnData,
    layer: str | None = None,
    genes: np.ndarray | list[str] | None = None,
    max_genes: int = 200,
    max_cells: int = 2000,
    seed: int = 0,
    selection: Literal["mean", "hvg"] = "hvg",
    metacell_size: int = 10,
    counts_layer: str | None = None,
    target_sum: float = 1e4,
    assert_log_normalized: bool = True,
) -> dict[str, Any]:
    """
    Compute pairwise Pearson and Kendall-tau gene-gene correlations.

    Kendall's tau and full pairwise correlation are quadratic in the number of
    genes, so genes (and optionally cells) are subsampled for tractability.

    When ``metacell_size > 0``, cells are aggregated into log-normalized metacell
    profiles (from raw counts) before correlating, which denoises single-cell
    measurement noise and better reveals co-expression structure. In that mode
    ``max_cells`` caps the number of metacells and ``layer`` is ignored (counts
    are read from ``counts_layer``).

    Args:
        adata: AnnData whose matrix (``.X`` or ``layer``) holds counts.
        layer: Optional layer name; ``None`` uses ``.X`` (single-cell mode).
        genes: Explicit genes to use. If ``None``, genes are selected according
            to ``selection``.
        max_genes: Maximum number of genes when ``genes`` is not provided.
        max_cells: Maximum number of cells (or metacells) used for correlation.
        seed: RNG seed for cell/metacell subsampling.
        selection: Gene ranking when ``genes`` is ``None``: ``"mean"`` (highest
            mean expression) or ``"hvg"`` (Seurat v3 highly variable genes).
        metacell_size: Cells per metacell; ``0`` uses single cells.
        counts_layer: Raw-counts layer for metacell aggregation; ``None`` uses
            ``.X`` (falling back to ``layers['counts']``).
        target_sum: Per-metacell normalization total.
        assert_log_normalized: When ``True``, assert the correlated data is
            log-normalized (not raw counts).

    Returns:
        Dict with:
        - ``"genes"``: gene names used (ordered).
        - ``"pearson"``, ``"kendall"``: ``(k, k)`` correlation matrices.
        - ``"pearson_values"``, ``"kendall_values"``: flattened upper triangles.
    """
    rng = np.random.default_rng(seed)
    gene_names = np.asarray(adata.var_names, dtype=str)

    if genes is None:
        selected_idx = _select_gene_indices(
            adata, layer=layer, max_genes=max_genes, selection=selection
        )
    else:
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        missing = [g for g in np.asarray(genes, dtype=str) if g not in gene_to_idx]
        if missing:
            raise KeyError(f"Genes not found in var_names: {missing[:10]}")
        selected_idx = np.array([gene_to_idx[str(g)] for g in genes], dtype=int)

    sub = adata[:, selected_idx]
    if metacell_size > 0:
        full_counts = _resolve_counts_matrix(adata, counts_layer)
        dense = _build_metacells(
            full_counts, selected_idx, metacell_size, max_cells, rng, target_sum
        )
    else:
        data = _as_csr(get_matrix(sub, layer))
        dense = data.toarray() if sparse.issparse(data) else np.asarray(data, dtype=np.float64)
        if dense.shape[0] > max_cells:
            row_idx = rng.choice(dense.shape[0], size=max_cells, replace=False)
            dense = dense[row_idx, :]

    if assert_log_normalized:
        _assert_log_normalized(dense)

    k = dense.shape[1]
    # Pearson: vectorized across all gene pairs.
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = np.corrcoef(dense, rowvar=False)
    pearson = np.nan_to_num(np.atleast_2d(pearson), nan=0.0)

    # Kendall tau: computed per pair (no vectorized form in SciPy).
    kendall = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            tau, _ = kendalltau(dense[:, i], dense[:, j])
            tau = 0.0 if np.isnan(tau) else float(tau)
            kendall[i, j] = tau
            kendall[j, i] = tau

    return {
        "genes": gene_names[selected_idx],
        "pearson": pearson,
        "kendall": kendall,
        "pearson_values": _upper_triangle(pearson),
        "kendall_values": _upper_triangle(kendall),
    }


def _median_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """
    Return the median and a percentile bootstrap CI, memory-safe for large n.

    The bootstrap loops over resamples instead of allocating an
    ``(n_boot, n)`` array, so it is safe for cell-wise tables with many rows.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    median = float(np.median(v))
    if v.size == 1:
        return median, median, median, 1

    rng = np.random.default_rng(seed)
    boot_medians = np.empty(n_boot, dtype=np.float64)
    n = v.size
    for i in range(n_boot):
        boot_medians[i] = np.median(v[rng.integers(0, n, size=n)])
    alpha = 1.0 - confidence
    ci_low = float(np.quantile(boot_medians, alpha / 2.0))
    ci_high = float(np.quantile(boot_medians, 1.0 - alpha / 2.0))
    return median, ci_low, ci_high, n


def summarize_statistics(
    stats: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Summarize a per-gene or per-cell statistic table with median and bootstrap CI.

    Collapses a table from ``gene_wise_statistics`` or ``cell_wise_statistics``
    into one median (with confidence interval) per statistic, giving
    text-reportable numbers for a single dataset.

    Args:
        stats: Per-gene or per-cell DataFrame of statistics.
        metrics: Columns to summarize. Defaults to all numeric columns.
        confidence: Confidence level for each interval.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for the bootstrap.

    Returns:
        DataFrame with one row per statistic and columns
        ``["statistic", "median", "ci_low", "ci_high", "n", "confidence"]``.
    """
    if metrics is None:
        selected = list(stats.select_dtypes(include=[np.number]).columns)
    else:
        selected = list(metrics)

    rows: list[dict[str, Any]] = []
    for metric in selected:
        if metric not in stats.columns:
            raise KeyError(f"Statistic {metric!r} not found in stats columns.")
        median, ci_low, ci_high, n = _median_ci(
            stats[metric].to_numpy(),
            confidence=confidence,
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {
                "statistic": metric,
                "median": median,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["statistic", "median", "ci_low", "ci_high", "n", "confidence"],
    )
