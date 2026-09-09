"""
TRADE-style perturbation-effect statistics for simulator validation.

This module estimates, per perturbation, the distribution of true differential
expression effect sizes and its transcriptome-wide impact, following TRADE
(Nadig et al., Nature Genetics 2025).

Pipeline:

1. ``pseudobulk_replicates`` aggregates single cells into pseudobulk samples with
    either real batches or random cell partitions per perturbation (and optional
    context), producing the replicate structure DESeq2 requires.
2. ``deseq2_effect_sizes`` fits one PyDESeq2 model across all conditions in a
    context, then computes each perturbation-versus-control Wald contrast and
    returns per-gene log2 fold changes and their standard errors.
3. ``transcriptome_wide_impact`` fits a zero-centered normal-mixture prior to
    deconvolve estimation noise, then reports the true effect-size variance and
    the effective number of differentially expressed genes (``pi_deg``).

When biological batches are available, they define pseudobulk replicates and
are included as a blocking factor in the DESeq2 design. Otherwise, cells are
randomly split into pseudo-replicates.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy import sparse
from scipy.stats import norm

from perturbations.util.anndata_util import get_matrix


def _deseq2_n_cpus() -> int | None:
    """
    Read the ``DESEQ2_N_CPUS`` override for PyDESeq2 worker count.

    Affects only how many loky workers PyDESeq2 spawns, never the numerics.
    Capping it avoids the /dev/shm memmap blow-up that OOMs large-core nodes.
    """
    raw = os.environ.get("DESEQ2_N_CPUS", "").strip()
    return int(raw) if raw else None


def _deseq2_size_factors_fit_type() -> str | None:
    """
    Read the ``DESEQ2_SIZE_FACTORS_FIT_TYPE`` override for PyDESeq2 normalization.

    Unlike ``DESEQ2_N_CPUS`` this DOES change the numerics, so it is opt-in and
    unset by default (PyDESeq2 then uses its own default, ``"ratio"``).

    The reason to override is that ``"ratio"`` (median-of-ratios) needs at least
    one gene that is non-zero in *every* pseudobulk sample. On a sparse
    perturbation panel -- one sample per (perturbation x batch), a few cells each
    -- no gene satisfies that, and PyDESeq2 silently falls back to ``"iterative"``,
    which optimizes one size factor per sample with derivative-free Powell and is
    computationally intractable at this scale. ``"poscounts"`` is DESeq2's
    estimator for exactly this case: it uses each sample's positive counts only,
    is closed-form, and never triggers the fallback.

    Accepted values: ``"ratio"``, ``"poscounts"``, ``"iterative"``.
    """
    raw = os.environ.get("DESEQ2_SIZE_FACTORS_FIT_TYPE", "").strip()
    if not raw:
        return None
    allowed = {"ratio", "poscounts", "iterative"}
    if raw not in allowed:
        raise ValueError(
            f"DESEQ2_SIZE_FACTORS_FIT_TYPE must be one of {sorted(allowed)}, got {raw!r}."
        )
    return raw


def _counts_matrix(adata: AnnData, layer: str | None) -> sparse.csr_matrix | np.ndarray:
    """Return the counts matrix from ``layer`` (or ``.X``) as CSR or 2D array."""
    matrix = get_matrix(adata, layer)
    if sparse.issparse(matrix):
        return matrix.tocsr()
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D counts matrix, got shape {arr.shape}.")
    return arr


def pseudobulk_replicates(
    adata: AnnData,
    perturbation_key: str = "perturbation",
    context_key: str | None = None,
    batch_key: str | None = None,
    layer: str | None = None,
    n_replicates: int = 3,
    min_cells_per_replicate: int = 10,
    seed: int = 0,
) -> AnnData:
    """
    Aggregate cells into pseudobulk samples with replicates.

    Within each ``(perturbation[, context])`` group, replicates are formed either
    from a provided ``batch_key`` (one pseudobulk per batch, using real replicate
    structure) or, when no batch is given, by randomly splitting cells into up to
    ``n_replicates`` partitions. Counts are summed per replicate, giving the
    structure DESeq2 needs. In the random-split path, replicates below
    ``min_cells_per_replicate`` cells are avoided; in the batch path, batches are
    used as-is. Groups that cannot form at least two replicates are dropped.

    Args:
        adata: AnnData with raw integer counts in ``.X`` or ``layer``.
        perturbation_key: ``obs`` column with perturbation labels.
        context_key: Optional ``obs`` column defining separate contexts; groups
            are formed per ``(perturbation, context)`` when provided.
        batch_key: Optional ``obs`` column of real batch/replicate labels. When
            provided, each batch within a group becomes one pseudobulk replicate
            and ``n_replicates`` is ignored.
        layer: Layer holding raw counts; ``None`` uses ``.X``.
        n_replicates: Target number of random pseudo-replicates per group (used
            only when ``batch_key`` is ``None``).
        min_cells_per_replicate: Minimum cells per replicate for the random-split
            path only; ignored when ``batch_key`` is provided.
        seed: RNG seed for the random cell partitioning.

    Returns:
        AnnData of summed counts with shape ``(n_samples, n_genes)``. ``obs`` has
        columns ``perturbation_key``, ``"replicate"``, ``"n_cells"`` (and
        ``context_key`` when provided); ``var`` is inherited from ``adata``.
    """
    rng = np.random.default_rng(seed)
    counts = _counts_matrix(adata, layer)
    obs = adata.obs
    pert_labels = np.asarray(obs[perturbation_key]).astype(str)
    batch_labels = np.asarray(obs[batch_key]).astype(str) if batch_key is not None else None

    # n_replicates only governs the random-split path; batches define their own count.
    if batch_labels is None and n_replicates < 2:
        raise ValueError(f"n_replicates must be >= 2 for DESeq2, got {n_replicates}.")

    if context_key is not None:
        ctx_labels = np.asarray(obs[context_key]).astype(str)
        group_ids = np.array([f"{p}||{c}" for p, c in zip(pert_labels, ctx_labels, strict=True)])
    else:
        ctx_labels = None
        group_ids = pert_labels

    sample_vectors: list[np.ndarray] = []
    sample_meta: list[dict[str, Any]] = []

    for group in np.unique(group_ids):
        cell_idx = np.where(group_ids == group)[0]

        if batch_labels is None:
            # Random pseudo-replicates: split cells into up to n_replicates chunks.
            k = min(n_replicates, cell_idx.size // max(1, min_cells_per_replicate))
            if k < 2:
                continue
            shuffled = rng.permutation(cell_idx)
            chunk_list: list[np.ndarray] = list(np.array_split(shuffled, k))
            replicate_ids: list[Any] = list(range(len(chunk_list)))
        else:
            # Real replicates: one pseudobulk per batch, used as-is (no size filter).
            group_batches = batch_labels[cell_idx]
            chunk_list = []
            replicate_ids = []
            for batch in np.unique(group_batches):
                batch_idx = cell_idx[group_batches == batch]
                # if batch_idx.size < min_cells_per_replicate:
                #     continue
                chunk_list.append(batch_idx)
                replicate_ids.append(batch)
            if len(chunk_list) < 2:
                continue

        pert_value = pert_labels[cell_idx[0]]
        ctx_value = ctx_labels[cell_idx[0]] if ctx_labels is not None else None

        for rep, chunk in zip(replicate_ids, chunk_list, strict=True):
            summed = counts[chunk, :].sum(axis=0)
            summed = np.asarray(summed).ravel()
            sample_vectors.append(summed)
            meta: dict[str, Any] = {
                perturbation_key: pert_value,
                "replicate": rep,
                "n_cells": int(chunk.size),
            }
            if context_key is not None:
                meta[context_key] = ctx_value
            sample_meta.append(meta)

    if not sample_vectors:
        raise ValueError(
            "No group could form at least two replicates. Lower "
            "min_cells_per_replicate, lower n_replicates, or check batch_key."
        )

    matrix = np.vstack(sample_vectors)
    matrix = np.rint(matrix).astype(np.int64, copy=False)  # DESeq2 expects integer counts
    meta_df = pd.DataFrame(sample_meta)
    sample_names = [
        f"{row[perturbation_key]}"
        + (f"|{row[context_key]}" if context_key is not None else "")
        + f"|rep{row['replicate']}"
        for _, row in meta_df.iterrows()
    ]
    meta_df.index = pd.Index(sample_names, name="sample")

    pseudobulk = AnnData(X=matrix, obs=meta_df, var=adata.var.copy())
    return pseudobulk


def _wald_chunk(
    design_matrix: np.ndarray,
    lfc_chunk: np.ndarray,
    disp_chunk: np.ndarray,
    size_factors: np.ndarray,
    contrast_vectors: np.ndarray,
    ridge_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Wald standard errors and statistics for a block of genes.

    Mirrors ``pydeseq2.utils.wald_test`` operation for operation, but builds the
    covariance matrix ``M`` and its regularized inverse ``H`` once per gene and
    reuses them across every contrast, instead of rebuilding both for each
    (gene, contrast) pair.

    Args:
        design_matrix: Design matrix of shape ``(n_samples, n_params)``.
        lfc_chunk: Fitted natural-log LFCs, shape ``(n_genes, n_params)``.
        disp_chunk: Per-gene dispersions, shape ``(n_genes,)``.
        size_factors: Per-sample size factors, shape ``(n_samples,)``.
        contrast_vectors: Contrast vectors, shape ``(n_params, n_contrasts)``.
        ridge_factor: Regularization matrix, shape ``(n_params, n_params)``.

    Returns:
        Tuple ``(se, stat)``, each of shape ``(n_genes, n_contrasts)``.
    """
    n_genes = lfc_chunk.shape[0]
    n_contrasts = contrast_vectors.shape[1]
    se = np.full((n_genes, n_contrasts), np.nan, dtype=np.float64)
    stat = np.full((n_genes, n_contrasts), np.nan, dtype=np.float64)

    # mu = exp(X @ LFC.T) scaled by size factors, as in DeseqStats.run_wald_test.
    mu_chunk = np.exp(design_matrix @ lfc_chunk.T) * size_factors[:, None]

    for gene in range(n_genes):
        mu = mu_chunk[:, gene]
        if not np.all(np.isfinite(mu)):
            # Genes with unfitted (all-zero) LFCs stay NaN, matching PyDESeq2.
            continue
        weights = mu / (1.0 + mu * disp_chunk[gene])
        cov = (design_matrix.T * weights[None, :]) @ design_matrix
        hat = np.linalg.inv(cov + ridge_factor)
        lfc = lfc_chunk[gene]
        for contrast_idx in range(n_contrasts):
            contrast = contrast_vectors[:, contrast_idx]
            hat_contrast = hat @ contrast
            wald_se = np.sqrt(hat_contrast.T @ cov @ hat_contrast)
            se[gene, contrast_idx] = wald_se
            stat[gene, contrast_idx] = contrast @ lfc / wald_se

    return se, stat


def _wald_tests_all_contrasts(
    dds: DeseqDataSet,
    contrast_vectors: np.ndarray,
    n_cpus: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run every Wald contrast against a fitted ``DeseqDataSet`` in one pass.

    PyDESeq2 exposes only a one-contrast-at-a-time API (``DeseqStats``), which
    rebuilds the per-gene covariance matrix and its inverse for every contrast
    even though both are contrast-independent once ``dds.deseq2()`` has run.
    For a Replogle-scale panel that is ~1000 redundant rebuilds per gene and
    dominates total runtime. This computes them once per gene instead.

    Args:
        dds: A ``DeseqDataSet`` on which ``deseq2()`` has already been run.
        contrast_vectors: Contrast vectors, shape ``(n_params, n_contrasts)``.
        n_cpus: Number of worker processes. ``None`` uses all available cores.

    Returns:
        Tuple ``(se, stat, pvalue)``, each of shape ``(n_genes, n_contrasts)``.
    """
    design_matrix = np.asarray(dds.obsm["design_matrix"].values, dtype=np.float64)
    lfc = np.asarray(dds.varm["LFC"].values, dtype=np.float64)
    dispersions = np.asarray(dds.var["dispersions"].values, dtype=np.float64)
    size_factors = np.asarray(dds.obs["size_factors"].values, dtype=np.float64)

    n_params = design_matrix.shape[1]
    if contrast_vectors.shape[0] != n_params:
        raise ValueError(
            f"Contrast vectors have {contrast_vectors.shape[0]} rows but the design "
            f"matrix has {n_params} columns."
        )
    # PyDESeq2 uses this fixed ridge when no LFC prior variance is supplied.
    ridge_factor = np.diag(np.repeat(1e-6, n_params))

    n_genes = lfc.shape[0]
    workers = max(1, min(n_cpus or multiprocessing.cpu_count(), n_genes))
    bounds = np.array_split(np.arange(n_genes), workers)

    if workers == 1:
        se, stat = _wald_chunk(
            design_matrix, lfc, dispersions, size_factors, contrast_vectors, ridge_factor
        )
    else:
        se = np.empty((n_genes, contrast_vectors.shape[1]), dtype=np.float64)
        stat = np.empty_like(se)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(
                    _wald_chunk,
                    design_matrix,
                    lfc[idx],
                    dispersions[idx],
                    size_factors,
                    contrast_vectors,
                    ridge_factor,
                ): idx
                for idx in bounds
                if idx.size
            }
            for future in as_completed(futures):
                idx = futures[future]
                chunk_se, chunk_stat = future.result()
                se[idx] = chunk_se
                stat[idx] = chunk_stat

    with np.errstate(invalid="ignore"):
        pvalue = 2.0 * norm.sf(np.abs(stat))
    pvalue[~np.isfinite(stat)] = np.nan
    return se, stat, pvalue


def deseq2_effect_sizes(
    pseudobulk: AnnData,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    batch_key: str | None = None,
    quiet: bool = True,
    n_cpus: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run PyDESeq2 for each perturbation versus control on pseudobulk samples.

    A single ``DeseqDataSet`` is fit across all conditions (sharing the dispersion
    trend), then a Wald contrast is computed per perturbation against control.
    When ``batch_key`` is provided and has at least two levels, it is included as
    a blocking factor (design ``~batch + condition``).

    Args:
        pseudobulk: AnnData of integer pseudobulk counts, e.g. from
            ``pseudobulk_replicates``, with at least two replicates per condition.
        perturbation_key: ``obs`` column with condition labels.
        control_label: Label of the control condition.
        batch_key: Optional ``obs`` column to block on in the design (e.g. real
            batches). Ignored when it has fewer than two levels.
        quiet: Whether to suppress PyDESeq2 progress output.
        n_cpus: Number of CPUs PyDESeq2 may use. ``None`` uses its default.

    Returns:
        Mapping ``perturbation -> DataFrame`` indexed by gene with columns
        ``["log2FoldChange", "lfcSE", "pvalue", "padj"]``.
    """
    labels = np.asarray(pseudobulk.obs[perturbation_key]).astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    if control_label not in unique_labels:
        raise ValueError(f"control_label {control_label!r} not found in {perturbation_key!r}.")

    # Sanitize labels so the design formula and contrasts are formula-safe
    # (perturbation names can contain '+', spaces, etc.).
    safe = {lab: f"c{i}" for i, lab in enumerate(unique_labels)}
    counts_df = pd.DataFrame(
        np.asarray(pseudobulk.X, dtype=np.int64),
        index=np.asarray(pseudobulk.obs_names, dtype=str),
        columns=np.asarray(pseudobulk.var_names, dtype=str),
    )
    metadata = pd.DataFrame(
        {"condition": [safe[lab] for lab in labels]},
        index=np.asarray(pseudobulk.obs_names, dtype=str),
    )

    # Optionally block on batch when it has at least two levels.
    design = "~condition"
    if batch_key is not None:
        batch_values = np.asarray(pseudobulk.obs[batch_key]).astype(str)
        unique_batches = list(dict.fromkeys(batch_values.tolist()))
        if len(unique_batches) >= 2:
            safe_batch = {b: f"b{i}" for i, b in enumerate(unique_batches)}
            metadata["batch"] = [safe_batch[b] for b in batch_values]
            design = "~batch + condition"

    # Require >= 2 replicates per condition for dispersion estimation.
    counts_per_condition = metadata["condition"].value_counts()
    valid_conditions = set(counts_per_condition[counts_per_condition >= 2].index)
    if safe[control_label] not in valid_conditions:
        raise ValueError("Control condition has fewer than 2 pseudo-replicates.")

    n_conditions = len(valid_conditions)
    n_batches = len(set(metadata["batch"])) if "batch" in metadata.columns else 0
    print(
        f"[{datetime.now():%H:%M:%S}] [deseq2] pid={os.getpid()} "
        f"samples={counts_df.shape[0]} genes={counts_df.shape[1]} "
        f"conditions={n_conditions} batches={n_batches} design='{design}' "
        f"n_cpus={n_cpus}",
        flush=True,
    )
    fit_start = time.time()
    size_factors_fit_type = _deseq2_size_factors_fit_type()
    if size_factors_fit_type is not None:
        print(
            f"[{datetime.now():%H:%M:%S}] [deseq2] pid={os.getpid()} "
            f"size_factors_fit_type={size_factors_fit_type} (overridden)",
            flush=True,
        )
    dds_kwargs: dict[str, Any] = {}
    if size_factors_fit_type is not None:
        dds_kwargs["size_factors_fit_type"] = size_factors_fit_type
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design=design,
        ref_level=["condition", safe[control_label]],
        quiet=quiet,
        n_cpus=n_cpus,
        **dds_kwargs,
    )
    dds.deseq2()
    print(
        f"[{datetime.now():%H:%M:%S}] [deseq2] pid={os.getpid()} "
        f"fit complete in {(time.time() - fit_start) / 60:.1f} min",
        flush=True,
    )

    contrast_targets = [
        lab for lab in unique_labels if lab != control_label and safe[lab] in valid_conditions
    ]
    n_contrasts = len(contrast_targets)
    contrast_start = time.time()

    # The Wald covariance terms (M and its inverse H) depend only on quantities
    # frozen by ``dds.deseq2()`` -- mu, dispersions, the design matrix and the
    # fitted LFCs -- so they are identical for every contrast. Computing them
    # once per gene instead of once per (gene, contrast) is the whole speedup.
    contrast_vectors = np.column_stack(
        [
            np.asarray(
                dds.contrast(
                    column="condition",
                    baseline=safe[control_label],
                    group_to_compare=safe[lab],
                ),
                dtype=np.float64,
            )
            for lab in contrast_targets
        ]
    )
    se_matrix, stat_matrix, pval_matrix = _wald_tests_all_contrasts(
        dds, contrast_vectors, n_cpus=n_cpus
    )
    print(
        f"[{datetime.now():%H:%M:%S}] [deseq2] pid={os.getpid()} "
        f"Wald tests for {n_contrasts} contrasts in "
        f"{(time.time() - contrast_start) / 60:.1f} min",
        flush=True,
    )

    # Cooks outlier flags are also contrast-independent, so evaluate them once
    # and disable DeseqStats' per-contrast recomputation.
    cooks_outliers = np.asarray(dds.cooks_outlier(), dtype=bool)

    post_start = time.time()
    results: dict[str, pd.DataFrame] = {}
    var_names = pd.Index(np.asarray(dds.var_names, dtype=str))
    for done, lab in enumerate(contrast_targets, start=1):
        stats = DeseqStats(
            dds,
            contrast=["condition", safe[lab], safe[control_label]],
            quiet=quiet,
            n_cpus=n_cpus,
        )
        # Inject the precomputed Wald results so ``summary`` skips the redundant
        # per-contrast refit and only runs the cheap p-value post-processing.
        pvalues = pval_matrix[:, done - 1].copy()
        pvalues[cooks_outliers] = np.nan
        stats.SE = pd.Series(se_matrix[:, done - 1], index=var_names)
        stats.statistics = pd.Series(stat_matrix[:, done - 1], index=var_names)
        stats.p_values = pd.Series(pvalues, index=var_names)
        stats.cooks_filter = False
        stats.summary()
        df = stats.results_df
        results[lab] = df.loc[:, ["log2FoldChange", "lfcSE", "pvalue", "padj"]].copy()
        if done % 50 == 0 or done == n_contrasts:
            rate = (time.time() - post_start) / done
            eta = (n_contrasts - done) * rate / 3600.0
            print(
                f"[{datetime.now():%H:%M:%S}] [deseq2] pid={os.getpid()} "
                f"contrasts {done}/{n_contrasts} ({rate:.1f}s each, ETA {eta:.1f}h)",
                flush=True,
            )

    return results


def _ash_autoselect_mixsd(
    betahat: np.ndarray, sebetahat: np.ndarray, mult: float = float(np.sqrt(2.0))
) -> np.ndarray:
    """
    Build a zero-centered normal-mixture standard-deviation grid (ashr-style).

    The grid spans a point mass at zero plus a geometric sequence from a small
    fraction of the smallest standard error up to twice the largest excess
    signal, matching ashr's ``autoselect.mixsd`` heuristic.
    """
    s_pos = sebetahat[sebetahat > 0]
    s_min = float(np.min(s_pos)) if s_pos.size else 1e-3
    sigmamin = s_min / 10.0
    excess = betahat**2 - sebetahat**2
    max_excess = float(np.max(excess)) if excess.size else 0.0
    sigmamax = 2.0 * np.sqrt(max_excess) if max_excess > 0.0 else 8.0 * sigmamin
    if sigmamax <= sigmamin:
        sigmamax = 8.0 * sigmamin
    n_steps = int(np.ceil(np.log(sigmamax / sigmamin) / np.log(mult)))
    grid = sigmamin * mult ** np.arange(n_steps + 1)
    return np.concatenate([[0.0], grid])


def _fit_ash_normal_mixture(
    betahat: np.ndarray,
    sebetahat: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a zero-centered normal-mixture prior by empirical Bayes (EM).

    Implements the normal-mixture variant of adaptive shrinkage (ashr): the
    true effects share a unimodal prior at zero, ``betahat ~ N(beta, se^2)``,
    and the mixture weights are fit by maximum likelihood via EM.

    Returns:
        Tuple ``(weights, component_variances)`` for the fitted prior.
    """
    sigma_grid = _ash_autoselect_mixsd(betahat, sebetahat)
    var_comp = sigma_grid**2
    total_var = np.maximum(sebetahat[:, None] ** 2 + var_comp[None, :], 1e-12)
    log_lik = -0.5 * (np.log(2.0 * np.pi * total_var) + betahat[:, None] ** 2 / total_var)

    weights = np.full(var_comp.size, 1.0 / var_comp.size)
    for _ in range(max_iter):
        log_post = np.log(weights + 1e-300)[None, :] + log_lik
        log_post -= log_post.max(axis=1, keepdims=True)
        resp = np.exp(log_post)
        resp /= resp.sum(axis=1, keepdims=True)
        new_weights = resp.mean(axis=0)
        new_weights /= new_weights.sum()
        if np.max(np.abs(new_weights - weights)) < tol:
            weights = new_weights
            break
        weights = new_weights

    return weights, var_comp


def _effective_n_deg(
    weights: np.ndarray,
    var_comp: np.ndarray,
    impact: float,
    n_genes: int,
) -> float:
    """
    Effective number of differentially expressed genes (TRADE ``pi_DEG``).

    TRADE defines ``pi_DEG = 3 * M / kappa``, where ``M`` is the number of genes
    and ``kappa`` is the kurtosis of the inferred effect-size distribution ``g``.
    A large effect on few genes gives large ``kappa`` and small ``pi_DEG``; a
    normal ``g`` gives ``kappa = 3`` and ``pi_DEG = M`` (all genes affected).

    For the zero-centered normal mixture ``g`` fit here (mean 0),
    ``kappa = 3 * sum(w * sigma^4) / impact^2`` with ``impact = sum(w * sigma^2)``,
    so ``pi_DEG = M * impact^2 / sum(w * sigma^4)``.

    Args:
        weights: Mixture weights of the fitted prior ``g``.
        var_comp: Component variances (``sigma^2``) of the fitted prior ``g``.
        impact: Variance of ``g`` (``sum(w * sigma^2)``), the transcriptome-wide
            impact.
        n_genes: Number of genes used to fit ``g``.

    Returns:
        The effective number of DE genes, or ``0.0`` when ``g`` collapses to a
        point mass at zero (no inferred effects).
    """
    fourth_moment_scale = float(np.sum(weights * var_comp**2))  # sum(w * sigma^4)
    if fourth_moment_scale <= 0.0:
        return 0.0
    return float(n_genes * impact**2 / fourth_moment_scale)


def transcriptome_wide_impact(
    lfc: np.ndarray,
    lfc_se: np.ndarray,
) -> dict[str, float]:
    """
    Estimate the variance of the true effect-size distribution (TRADE impact).

    Uses empirical-Bayes deconvolution with a zero-centered normal mixture prior
    (the ashr approach TRADE uses): the transcriptome-wide impact is the variance
    of the fitted prior, i.e. the variance of the true log2-fold-change
    distribution net of estimation noise. The effective number of differentially
    expressed genes (``pi_deg``) is derived from the kurtosis of the same prior.

    Args:
        lfc: Per-gene log2 fold changes for one perturbation.
        lfc_se: Per-gene standard errors of ``lfc``.

    Returns:
        Dict with ``transcriptome_wide_impact`` (deconvolved variance),
        ``sqrt_transcriptome_wide_impact`` (its square root), ``pi_deg``
        (effective number of DE genes), ``observed_variance``,
        ``mean_sampling_variance`` and ``n_genes``.
    """
    lfc = np.asarray(lfc, dtype=np.float64)
    lfc_se = np.asarray(lfc_se, dtype=np.float64)
    finite = np.isfinite(lfc) & np.isfinite(lfc_se)
    lfc = lfc[finite]
    lfc_se = lfc_se[finite]

    if lfc.size == 0:
        return {
            "transcriptome_wide_impact": float("nan"),
            "sqrt_transcriptome_wide_impact": float("nan"),
            "pi_deg": float("nan"),
            "observed_variance": float("nan"),
            "mean_sampling_variance": float("nan"),
            "n_genes": 0.0,
        }

    observed_variance = float(np.var(lfc, ddof=0))
    mean_sampling_variance = float(np.mean(lfc_se**2))

    if lfc.size < 2:
        # Too few genes to fit a mixture; fall back to a clipped moment estimate.
        impact = max(observed_variance - mean_sampling_variance, 0.0)
        pi_deg = float("nan")
    else:
        location = float(np.mean(lfc))
        weights, var_comp = _fit_ash_normal_mixture(lfc - location, lfc_se)
        impact = float(np.sum(weights * var_comp))
        pi_deg = _effective_n_deg(weights, var_comp, impact, lfc.size)

    return {
        "transcriptome_wide_impact": impact,
        "sqrt_transcriptome_wide_impact": float(np.sqrt(impact)),
        "pi_deg": pi_deg,
        "observed_variance": observed_variance,
        "mean_sampling_variance": mean_sampling_variance,
        "n_genes": float(lfc.size),
    }


def _perturbation_effect_rows_for_context(
    pseudobulk: AnnData,
    context: str | None,
    perturbation_key: str,
    control_label: str,
    context_key: str | None,
    block_on_replicate: bool,
    deg_fdr: float,
    quiet: bool,
    n_cpus: int | None,
) -> list[dict[str, Any]]:
    """Compute all perturbation-effect rows for one context."""
    effects = deseq2_effect_sizes(
        pseudobulk,
        perturbation_key=perturbation_key,
        control_label=control_label,
        batch_key="replicate" if block_on_replicate else None,
        quiet=quiet,
        n_cpus=n_cpus,
    )
    rows: list[dict[str, Any]] = []
    for pert, df in effects.items():
        lfc = df["log2FoldChange"].to_numpy()
        lfc_se = df["lfcSE"].to_numpy()
        padj = df["padj"].to_numpy()
        impact = transcriptome_wide_impact(lfc, lfc_se)
        row: dict[str, Any] = {
            perturbation_key: pert,
            "transcriptome_wide_impact": impact["transcriptome_wide_impact"],
            "sqrt_transcriptome_wide_impact": impact["sqrt_transcriptome_wide_impact"],
            "pi_deg": impact["pi_deg"],
            "observed_variance": impact["observed_variance"],
            "n_deg": int(np.nansum(padj < deg_fdr)),
            "mean_abs_lfc": float(np.nanmean(np.abs(lfc))) if lfc.size else float("nan"),
            "n_genes": int(impact["n_genes"]),
        }
        if context_key is not None:
            row[context_key] = context
        rows.append(row)
    return rows


def _report_context_progress(
    completed: int,
    total: int,
    context_key: str | None,
    context: str | None,
) -> None:
    """Print a timestamped TRADE context-completion update."""
    key = context_key or "context"
    value = context if context is not None else "all"
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] TRADE: context "
        f"{completed}/{total} complete ({key}={value})",
        flush=True,
    )


def perturbation_effect_statistics(
    adata: AnnData,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    context_key: str | None = None,
    batch_key: str | None = None,
    layer: str | None = None,
    n_replicates: int = 3,
    min_cells_per_replicate: int = 10,
    deg_fdr: float = 0.05,
    seed: int = 0,
    quiet: bool = True,
    context_workers: int = 1,
) -> pd.DataFrame:
    """
    Compute per-perturbation effect-size statistics for one dataset.

    Builds pseudobulk pseudo-replicates, runs PyDESeq2 per perturbation versus
    control (within each context when ``context_key`` is given), and summarizes
    each perturbation's effect with its TRADE transcriptome-wide impact, the
    effective number of DE genes (``pi_deg``, from the inferred distribution's
    kurtosis), the significance-thresholded DEG count (``n_deg``), and mean
    absolute log2 fold change.

    Args:
        adata: AnnData with raw integer counts in ``.X`` or ``layer``.
        perturbation_key: ``obs`` column with perturbation labels.
        control_label: Label of the control condition.
        context_key: Optional ``obs`` column; effects are estimated within each
            context separately when provided.
        batch_key: Optional ``obs`` column of real batch/replicate labels passed
            to ``pseudobulk_replicates``; when given, batches define replicates.
        layer: Layer holding raw counts; ``None`` uses ``.X``.
        n_replicates: Target pseudo-replicates per group.
        min_cells_per_replicate: Minimum cells per pseudo-replicate.
        deg_fdr: FDR threshold for counting differentially expressed genes.
        seed: RNG seed for pseudo-replicate partitioning.
        quiet: Whether to suppress PyDESeq2 output.
        context_workers: Maximum context-level worker processes. Values greater
            than one run contexts in parallel and restrict each PyDESeq2 fit to
            one CPU to avoid nested process pools.

    Returns:
        DataFrame with one row per perturbation (per context) and columns
        ``[perturbation_key, (context_key,) "transcriptome_wide_impact",
        "sqrt_transcriptome_wide_impact", "pi_deg", "observed_variance",
        "n_deg", "mean_abs_lfc", "n_genes"]``.
    """
    if context_workers < 1:
        raise ValueError(f"context_workers must be >= 1, got {context_workers}.")

    pseudobulk = pseudobulk_replicates(
        adata,
        perturbation_key=perturbation_key,
        context_key=context_key,
        batch_key=batch_key,
        layer=layer,
        n_replicates=n_replicates,
        min_cells_per_replicate=min_cells_per_replicate,
        seed=seed,
    )

    if context_key is None:
        contexts: list[Any] = [None]
    else:
        contexts = list(np.unique(np.asarray(pseudobulk.obs[context_key]).astype(str)))

    context_data: list[tuple[str | None, AnnData]] = []
    for ctx in contexts:
        sub = (
            pseudobulk
            if ctx is None
            else pseudobulk[np.asarray(pseudobulk.obs[context_key]).astype(str) == ctx].copy()
        )
        context_data.append((ctx, sub))

    worker_count = min(context_workers, len(context_data))
    common_args = (
        perturbation_key,
        control_label,
        context_key,
        batch_key is not None,
        deg_fdr,
        quiet,
    )
    row_groups: list[list[dict[str, Any]]] = [[] for _ in context_data]
    if worker_count == 1:
        for index, (ctx, sub) in enumerate(context_data):
            row_groups[index] = _perturbation_effect_rows_for_context(
                sub, ctx, *common_args, _deseq2_n_cpus()
            )
            _report_context_progress(index + 1, len(context_data), context_key, ctx)
    else:
        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=spawn_context) as executor:
            future_contexts = {
                executor.submit(
                    _perturbation_effect_rows_for_context,
                    sub,
                    ctx,
                    *common_args,
                    1,
                ): (index, ctx)
                for index, (ctx, sub) in enumerate(context_data)
            }
            for completed, future in enumerate(as_completed(future_contexts), start=1):
                index, ctx = future_contexts[future]
                row_groups[index] = future.result()
                _report_context_progress(completed, len(context_data), context_key, ctx)

    rows = [row for group in row_groups for row in group]

    return pd.DataFrame(rows)


def median_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Compute the median and a bootstrap confidence interval.

    The interval is a percentile bootstrap CI for the median, suitable for
    reporting a single dataset's typical value in text.

    Args:
        values: 1D array of per-perturbation values.
        confidence: Confidence level for the interval (e.g. ``0.95``).
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for the bootstrap.

    Returns:
        Dict with ``median``, ``ci_low``, ``ci_high`` and ``n``.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0.0}

    median = float(np.median(v))
    if v.size == 1:
        return {"median": median, "ci_low": median, "ci_high": median, "n": 1.0}

    rng = np.random.default_rng(seed)
    boot_medians = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    alpha = 1.0 - confidence
    return {
        "median": median,
        "ci_low": float(np.quantile(boot_medians, alpha / 2.0)),
        "ci_high": float(np.quantile(boot_medians, 1.0 - alpha / 2.0)),
        "n": float(v.size),
    }


def summarize_perturbation_statistics(
    stats: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Summarize per-perturbation statistics with a median and bootstrap CI.

    Collapses the per-perturbation table from ``perturbation_effect_statistics``
    into one median (with confidence interval) per metric, giving text-reportable
    numbers for a single dataset.

    Args:
        stats: Per-perturbation DataFrame, e.g. from
            ``perturbation_effect_statistics``.
        metrics: Metric columns to summarize. Defaults to the effect columns
            present among ``transcriptome_wide_impact``,
            ``sqrt_transcriptome_wide_impact``, ``pi_deg``,
            ``observed_variance``, ``n_deg`` and ``mean_abs_lfc``.
        confidence: Confidence level for each interval.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for the bootstrap.

    Returns:
        DataFrame with one row per metric and columns
        ``["metric", "median", "ci_low", "ci_high", "n_perturbations",
        "confidence"]``.
    """
    default_metrics = [
        "transcriptome_wide_impact",
        "sqrt_transcriptome_wide_impact",
        "pi_deg",
        "observed_variance",
        "n_deg",
        "mean_abs_lfc",
    ]
    selected = (
        [m for m in default_metrics if m in stats.columns] if metrics is None else list(metrics)
    )

    rows: list[dict[str, Any]] = []
    for metric in selected:
        if metric not in stats.columns:
            raise KeyError(f"Metric {metric!r} not found in stats columns.")
        summary = median_ci(
            stats[metric].to_numpy(),
            confidence=confidence,
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {
                "metric": metric,
                "median": summary["median"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "n_perturbations": int(summary["n"]),
                "confidence": confidence,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["metric", "median", "ci_low", "ci_high", "n_perturbations", "confidence"],
    )
