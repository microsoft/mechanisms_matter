"""Utility helpers for perturbation analysis calculations."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from ..data.preprocess import perturbation_targets_in_gene_set
from ..util.anndata_util import get_matrix
from .common import NORM_LAYER_KEY, label_to_target_tokens


def _to_vector(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


def build_perturbation_label_mapping(
    perturbation_labels: pd.Series | np.ndarray,
) -> dict[str, int]:
    """Build a stable non-control perturbation-to-id mapping."""
    labels = pd.Series(perturbation_labels).astype(str)
    if "control" not in set(labels.unique()):
        raise ValueError("Control label 'control' not found in 'perturbation'.")

    non_control_labels = sorted(label for label in labels.unique() if label != "control")
    return {label: idx for idx, label in enumerate(non_control_labels)}


def encode_perturbations(
    adata: ad.AnnData,
) -> tuple[ad.AnnData, dict[str, int]]:
    """Attach stable integer perturbation ids to a real-dataset AnnData object."""
    if "perturbation" not in adata.obs.columns:
        raise KeyError(
            f"Missing obs column 'perturbation'. Available columns: {list(adata.obs.columns)}"
        )

    perturbation_labels = adata.obs["perturbation"].astype(str)
    mapping = build_perturbation_label_mapping(perturbation_labels)

    encoded = np.full(adata.n_obs, -1, dtype=np.int32)
    non_control_mask = perturbation_labels.values != "control"
    encoded[non_control_mask] = np.asarray(
        [mapping[label] for label in perturbation_labels.values[non_control_mask]],
        dtype=np.int32,
    )

    adata.obs["perturbation"] = perturbation_labels.values
    adata.obs["perturbation_id"] = encoded
    return adata, mapping


def ensure_normalized_log1p_layer(
    adata: ad.AnnData,
    output_layer_key: str = NORM_LAYER_KEY,
    source_layer: str | None = "counts",
    target_sum: float = 1e4,
) -> None:
    """Build a normalized/log1p layer once when it is missing."""
    if output_layer_key in adata.layers:
        return

    if source_layer is not None:
        if source_layer not in adata.layers:
            raise KeyError(
                f"Requested source_layer='{source_layer}' not found. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        source = adata.layers[source_layer]
    else:
        source = adata.X

    tmp = ad.AnnData(X=source.copy())
    sc.pp.normalize_total(tmp, target_sum=float(target_sum))
    sc.pp.log1p(tmp)

    if sparse.issparse(tmp.X):
        adata.layers[output_layer_key] = tmp.X.astype(np.float32)
    else:
        adata.layers[output_layer_key] = np.asarray(tmp.X, dtype=np.float32)


def load_real_dataset(
    dataset_path: str,
) -> tuple[ad.AnnData, dict[str, int]]:
    """Load a real dataset from disk and attach encoded perturbation ids."""
    adata = ad.read_h5ad(dataset_path)
    return encode_perturbations(adata)


def validate_perturbation_targets_subset_from_obs(
    obs: pd.DataFrame,
    gene_names: pd.Index | np.ndarray,
    control_label: str,
) -> None:
    """Validate that every perturbation target token is present in the gene set."""
    gene_name_set = set(pd.Index(gene_names).astype(str))
    missing_targets = sorted(
        {
            token
            for label in obs["perturbation"].astype(str).unique()
            if label != control_label
            and not perturbation_targets_in_gene_set(
                label,
                gene_name_set,
                control_label=control_label,
            )
            for token in label_to_target_tokens(label)
            if token not in gene_name_set
        }
    )
    if not missing_targets:
        return

    preview = ", ".join(missing_targets[:10])
    suffix = "" if len(missing_targets) <= 10 else f", ... (+{len(missing_targets) - 10} more)"
    raise ValueError(
        "Found perturbation targets that are not present in adata.var_names. "
        f"Missing targets: {preview}{suffix}"
    )


def validate_perturbation_targets_subset(
    adata: ad.AnnData,
    control_label: str,
) -> None:
    """Validate perturbation targets against the genes present in an AnnData object."""
    validate_perturbation_targets_subset_from_obs(
        obs=adata.obs,
        gene_names=adata.var_names,
        control_label=control_label,
    )


def count_non_control_perturbations(
    obs: pd.DataFrame,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
) -> int:
    """Count unique non-control perturbations from observation metadata."""
    labels = pd.Index(obs[perturbation_key].astype(str).unique())
    return int((labels != str(control_label)).sum())


def compute_means_by_perturbation(
    adata_view,
    perturbation_ids: np.ndarray,
    layer_key: str | None,
    perturbation_key: str = "perturbation",
    missing_group_context: str | None = None,
) -> np.ndarray:
    """Compute per-perturbation mean expression vectors from an AnnData-like view."""
    means = np.empty((perturbation_ids.size, adata_view.n_vars), dtype=np.float32)
    labels = adata_view.obs[perturbation_key].to_numpy(copy=False)
    context_suffix = f" in {missing_group_context}" if missing_group_context else ""

    for idx, pert_id in enumerate(perturbation_ids):
        pert_mask = labels == pert_id
        if int(np.sum(pert_mask)) == 0:
            raise ValueError(f"No cells found for perturbation label {pert_id!r}{context_suffix}.")
        matrix = get_matrix(adata_view[pert_mask, :], layer_key=layer_key)
        means[idx, :] = _to_vector(matrix.mean(axis=0)).astype(np.float32, copy=False)
    return means


def build_perturbation_id_map(obs_df: pd.DataFrame) -> dict[str, int]:
    """Map synthetic perturbation labels back to their integer perturbation ids."""
    if "perturbation_id" not in obs_df.columns:
        raise KeyError("Expected 'perturbation_id' in synthetic obs metadata.")

    pairs = obs_df[["perturbation", "perturbation_id"]].drop_duplicates()
    return {
        str(label): int(pid)
        for label, pid in zip(
            pairs["perturbation"].to_numpy(),
            pairs["perturbation_id"].to_numpy(),
            strict=True,
        )
    }


def true_degs_for_context(
    dataset_name: str,
    affected_genes: Any,
    pert_label_to_id: dict[str, int],
    perturbation_ids: np.ndarray,
    context_axis: str | None = None,
    context_values: tuple[Any, ...] = (),
) -> list[np.ndarray]:
    """Return true DEG masks aligned to one evaluation bucket."""
    if dataset_name == "directDGP":
        affected_genes_for_eval = affected_genes
    elif dataset_name == "causalDGP":
        if context_axis != "cell_line":
            raise ValueError(
                "causalDGP true-DEG mapping requires evaluation buckets on the 'cell_line' axis. "
                f"Got {context_axis!r}."
            )
        cell_line_idx = int(context_values[0])
        affected_genes_for_eval = affected_genes[cell_line_idx]
    else:
        raise ValueError(
            f"Unsupported dataset_name for mapping affected_genes to eval perturbation ids: {dataset_name}"
        )

    return [affected_genes_for_eval[pert_label_to_id[str(pert_id)]] for pert_id in perturbation_ids]


def est_cost(params):
    """
    Estimate cost for the computation, based on the number of cells and genes.

    :param params: Dictionary containing parameters including G, N0, Nk, and P
    """
    G, N0, Nk, P = params["G"], params["N0"], params["Nk"], params["P"]
    rows = N0 + P * Nk
    return rows * G


def systematic_variation(
    ptb_shifts: np.ndarray,
    avg_ptb_shift: np.ndarray,
) -> float:
    """
    Calculate the average cosine similarity between perturbation-specific shifts.

    and the average perturbation effect.

    :param ptb_shifts: perturbation shifts matrix of shape (n_perturbations, n_genes)
    :param avg_ptb_shift: average perturbation shift vector of shape (n_genes,)
    """
    similarities = cosine_similarity(ptb_shifts, avg_ptb_shift.reshape(1, -1)).flatten()
    return float(np.mean(similarities))


def intra_correlation(
    ptb_shifts: np.ndarray,
) -> float:
    """Compute mean pairwise Pearson correlation across perturbation-specific shifts."""
    corr_matrix = np.corrcoef(ptb_shifts)
    lower_tri_indices = np.tril_indices(corr_matrix.shape[0], k=-1)
    mean_corr = np.mean(corr_matrix[lower_tri_indices])
    return float(mean_corr)


def sum_and_sumsq(matrix):
    """Return per-gene sums and squared sums for dense/sparse matrices."""
    if sparse.issparse(matrix):
        gene_sum = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
        gene_sumsq = (
            np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel().astype(np.float64, copy=False)
        )
        return gene_sum, gene_sumsq

    dense_matrix = np.asarray(matrix, dtype=np.float32)
    gene_sum = dense_matrix.sum(axis=0, dtype=np.float64)
    gene_sumsq = np.square(dense_matrix, dtype=np.float64).sum(axis=0, dtype=np.float64)
    return gene_sum, gene_sumsq
