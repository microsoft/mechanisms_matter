"""Utilities for extracting matrices and fitting PCA on AnnData-like objects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from anndata import AnnData
from anndata.experimental import AnnCollection
from scipy import sparse
from sklearn.decomposition import IncrementalPCA


# pyright: reportUnknownMemberType=false
def get_matrix(data_obj: Any, layer_key: str | None):
    """Return matrix from X or a requested layer."""
    if layer_key is None:
        return data_obj.X

    layers = getattr(data_obj, "layers", None)
    if layers is None:
        raise KeyError(f"Requested layer '{layer_key}' but no layers are available.")

    try:
        return layers[layer_key]
    except KeyError as e:
        available_layers = list(layers.keys()) if hasattr(layers, "keys") else []
        raise KeyError(
            f"Requested layer '{layer_key}' not found. Available layers: {available_layers}"
        ) from e


def materialize_adata(
    adata_like: Any,
    expression_layer: str | None = None,
) -> AnnData:
    """Materialize an AnnData-like object and optionally copy a layer into `.X`."""
    adata = adata_like.to_adata() if hasattr(adata_like, "to_adata") else adata_like.copy()

    if expression_layer is None:
        return adata

    source = get_matrix(adata, expression_layer)
    if sparse.issparse(source):
        adata.X = source.copy()
    else:
        adata.X = np.asarray(source, dtype=np.float32).copy()
    return adata


def extract_rows(
    data_obj: AnnData | AnnCollection,
    row_idx: np.ndarray,
    layer_key: str | None,
) -> np.ndarray:
    """Extract selected rows as a dense float64 matrix."""
    if row_idx.size == 0:
        return np.empty((0, int(data_obj.n_vars)), dtype=np.float64)

    view = data_obj[row_idx, :]
    matrix = get_matrix(view, layer_key)
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def obs_has_key(obs_obj: Any, key: str) -> bool:
    """Return True when an obs-like object contains the requested key."""
    if obs_obj is None:
        return False

    columns = getattr(obs_obj, "columns", None)
    if columns is not None:
        return key in columns

    keys_fn = getattr(obs_obj, "keys", None)
    if callable(keys_fn):
        try:
            keys_result = keys_fn()
            if isinstance(keys_result, Iterable):
                return key in set(keys_result)  # type: ignore
        except Exception:
            pass

    try:
        _ = obs_obj[key]
        return True
    except Exception:
        return False


def iterate_batches(
    data_obj: AnnData | AnnCollection,
    layer_key: str | None,
    batch_size: int,
    obs_key: str | None = "perturbation",
) -> Iterable[tuple[np.ndarray, np.ndarray | None]]:
    """
    Iterate rows in mini-batches.

    Yields:
        (batch_matrix, batch_obs_values) where batch_obs_values is None when
        obs_key is None.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    iterate_axis = getattr(data_obj, "iterate_axis", None)
    if callable(iterate_axis):
        for batch_view, _ in data_obj.iterate_axis(batch_size=batch_size, axis=0, shuffle=False):  # type: ignore
            matrix = get_matrix(batch_view, layer_key)
            if sparse.issparse(matrix):
                matrix = matrix.toarray()
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)

            obs_values = None
            if obs_key is not None:
                obs_values = np.asarray(batch_view.obs[obs_key])  # type: ignore
            yield matrix, obs_values
        return

    n_obs = int(data_obj.n_obs)
    for start in range(0, n_obs, batch_size):
        stop = min(start + batch_size, n_obs)
        batch_view = data_obj[start:stop, :]
        matrix = get_matrix(batch_view, layer_key)
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        obs_values = None
        if obs_key is not None:
            obs_values = np.asarray(batch_view.obs[obs_key])  # type: ignore
        yield matrix, obs_values


def fit_control_incremental_pca(
    data_obj: AnnData | AnnCollection,
    layer_key: str | None,
    control_label: str | int | float = -1,
    n_pca_components: int = 50,
    batch_size: int = 1024,
    obs_key: str = "perturbation",
    data_name: str = "data_obj",
) -> IncrementalPCA:
    """
    Fit IncrementalPCA on control cells only.

    Cells are selected by data_obj.obs[obs_key] == control_label.
    """
    if not obs_has_key(getattr(data_obj, "obs", None), obs_key):
        raise KeyError(f"{data_name}.obs must contain a '{obs_key}' column.")

    if int(n_pca_components) <= 0:
        raise ValueError(f"n_pca_components must be a positive integer. Got {n_pca_components}.")

    labels = np.asarray(data_obj.obs[obs_key])  # type: ignore
    n_control = int(np.sum(labels == control_label))
    if n_control <= 0:
        raise ValueError(
            f"No control cells found for control_label={control_label!r} in {data_name}."
        )

    n_components = min(int(n_pca_components), n_control, int(data_obj.n_vars))
    fit_batch_size = int(batch_size)
    if fit_batch_size < n_components:
        raise ValueError(
            f"batch_size ({fit_batch_size}) must be >= n_components "
            + f"({n_components}) for IncrementalPCA partial_fit."
        )

    pca_model = IncrementalPCA(n_components=n_components, batch_size=fit_batch_size)
    tail_batch: np.ndarray | None = None
    fitted_once = False

    for batch_matrix, batch_labels in iterate_batches(
        data_obj=data_obj,
        layer_key=layer_key,
        batch_size=fit_batch_size,
        obs_key=obs_key,
    ):
        if batch_labels is None:
            continue
        control_mask = batch_labels == control_label
        if not np.any(control_mask):
            continue

        control_batch = np.asarray(batch_matrix[control_mask, :], dtype=np.float64)
        if control_batch.shape[0] == 0:
            continue

        if control_batch.shape[0] < n_components:
            if tail_batch is None:
                tail_batch = control_batch
            else:
                tail_batch = np.concatenate([tail_batch, control_batch], axis=0)
            continue

        pca_model.partial_fit(control_batch)
        fitted_once = True

    # partial_fit requires at least n_components rows, so pad a tiny tail if needed.
    if tail_batch is not None and tail_batch.shape[0] > 0:
        if tail_batch.shape[0] < n_components:
            pad_needed = n_components - tail_batch.shape[0]
            pad = tail_batch[np.arange(pad_needed) % tail_batch.shape[0]]
            tail_batch = np.concatenate([tail_batch, pad], axis=0)
        pca_model.partial_fit(tail_batch)
        fitted_once = True

    if not fitted_once:
        raise ValueError("IncrementalPCA failed to fit on control cells.")

    return pca_model


def fit_incremental_pca_all_cells(
    data_obj: AnnData | AnnCollection,
    layer_key: str | None,
    n_pca_components: int = 50,
    batch_size: int = 1024,
    data_name: str = "data_obj",
) -> IncrementalPCA:
    """Fit IncrementalPCA on all cells."""
    if int(n_pca_components) <= 0:
        raise ValueError(f"n_pca_components must be a positive integer. Got {n_pca_components}.")

    n_obs = int(data_obj.n_obs)
    if n_obs <= 0:
        raise ValueError(f"{data_name} has no observations to fit IncrementalPCA.")

    n_components = min(int(n_pca_components), n_obs, int(data_obj.n_vars))
    fit_batch_size = int(batch_size)
    if fit_batch_size < n_components:
        raise ValueError(
            f"batch_size ({fit_batch_size}) must be >= n_components "
            + f"({n_components}) for IncrementalPCA partial_fit."
        )

    pca_model = IncrementalPCA(n_components=n_components, batch_size=fit_batch_size)
    tail_batch: np.ndarray | None = None
    fitted_once = False

    for batch_matrix, _ in iterate_batches(
        data_obj=data_obj,
        layer_key=layer_key,
        batch_size=fit_batch_size,
        obs_key=None,
    ):
        all_batch = np.asarray(batch_matrix, dtype=np.float64)
        if all_batch.shape[0] == 0:
            continue

        if all_batch.shape[0] < n_components:
            if tail_batch is None:
                tail_batch = all_batch
            else:
                tail_batch = np.concatenate([tail_batch, all_batch], axis=0)
            continue

        pca_model.partial_fit(all_batch)
        fitted_once = True

    if tail_batch is not None and tail_batch.shape[0] > 0:
        if tail_batch.shape[0] < n_components:
            pad_needed = n_components - tail_batch.shape[0]
            pad = tail_batch[np.arange(pad_needed) % tail_batch.shape[0]]
            tail_batch = np.concatenate([tail_batch, pad], axis=0)
        pca_model.partial_fit(tail_batch)
        fitted_once = True

    if not fitted_once:
        raise ValueError("IncrementalPCA failed to fit on all cells.")

    return pca_model
