"""Vendi-score metrics for reconstruction quality evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
from anndata import AnnData
from anndata.experimental import AnnCollection
from sklearn.decomposition import PCA

from perturbations.metrics.reconstruction.distance_util import (
    biased_mmd2_from_kernel_means,
    estimate_mmd_gamma,
    kernel_mean,
    pairwise_squared_distances,
)
from perturbations.util.anndata_util import (
    extract_rows,
    fit_control_incremental_pca,
    fit_incremental_pca_all_cells,
    iterate_batches,
    obs_has_key,
)


def _vendi_from_spectrum(eigenvalues: np.ndarray) -> float:
    """Vendi score from spectrum with numerical safety."""
    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = np.clip(eig, 0.0, None)
    eig_sum = float(eig.sum())
    if eig_sum <= 0.0:
        return 1.0

    eig /= eig_sum
    positive = eig[eig > 0.0]
    if positive.size == 0:
        return 1.0
    entropy: float = float(-np.sum(positive * np.log(positive)))  # type: ignore[reportUnknownArgumentType]
    return float(np.exp(entropy))


def _prepare_control_null_calibration(
    ac: AnnData | AnnCollection,
    layer_key: str,
    control_label: str,
    n_splits: int,
    null_quantile: float,
    target_null_similarity: float,
) -> tuple[np.ndarray, int]:
    """Validate null calibration and return control cells with a matched split size."""
    if int(n_splits) <= 0:
        raise ValueError(f"n_splits must be a positive integer. Got {n_splits}.")
    if not 0.0 < null_quantile < 1.0:
        raise ValueError(f"null_quantile must be between 0 and 1. Got {null_quantile}.")
    if not 0.0 < target_null_similarity < 1.0:
        raise ValueError(
            "target_null_similarity must be between 0 and 1. " + f"Got {target_null_similarity}."
        )
    if not obs_has_key(getattr(ac, "obs", None), "perturbation"):
        raise KeyError("ac.obs must contain a 'perturbation' column.")

    labels = np.asarray(ac.obs["perturbation"])  # type: ignore[reportUnknownArgumentType]
    control_indices = np.flatnonzero(labels == control_label)
    if control_indices.size < 2:
        raise ValueError(
            f"At least two control cells are required for control_label={control_label!r}."
        )
    perturbation_labels = labels[labels != control_label]
    if perturbation_labels.size == 0:
        raise ValueError("At least one non-control perturbation group is required.")
    _, perturbation_counts = np.unique(perturbation_labels, return_counts=True)
    split_size = min(int(np.median(perturbation_counts)), int(control_indices.size) // 2)
    if split_size <= 0:
        raise ValueError("Unable to form two non-empty disjoint control splits.")

    return extract_rows(ac, control_indices, layer_key), split_size


def _calibrate_outer_sigma_squared(
    null_distances_squared: np.ndarray,
    null_quantile: float,
    target_null_similarity: float,
) -> float:
    """Map a positive null-distance quantile to an outer RBF variance."""
    null_distance_squared = float(np.quantile(null_distances_squared, null_quantile))
    if not np.isfinite(null_distance_squared) or null_distance_squared <= 0.0:
        raise ValueError(
            "Null calibration produced no positive finite squared distance at "
            + f"quantile {null_quantile}."
        )
    return float(-null_distance_squared / (2.0 * np.log(target_null_similarity)))


def _iterate_control_splits(
    control_cells: np.ndarray,
    split_size: int,
    n_splits: int,
    random_state: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield reproducible disjoint control splits with matched sizes."""
    rng = np.random.default_rng(random_state)
    for _ in range(int(n_splits)):
        sampled_indices = rng.choice(
            control_cells.shape[0],
            size=2 * split_size,
            replace=False,
        )
        yield (
            control_cells[sampled_indices[:split_size]],
            control_cells[sampled_indices[split_size:]],
        )


def fit_vendi_pseudobulk_pca(
    pseudobulk: np.ndarray,
    control_idx: int | None = None,
    n_pca_components: int = 50,
    random_state: int = 0,
) -> PCA:
    """Fit PCA on observed perturbation pseudobulks, excluding an optional control row."""
    X = np.asarray(pseudobulk, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        raise ValueError("pseudobulk must contain at least two non-empty rows.")
    if not np.all(np.isfinite(X)):
        raise ValueError("pseudobulk contains non-finite values (NaN or inf).")
    if int(n_pca_components) <= 0:
        raise ValueError(f"n_pca_components must be a positive integer. Got {n_pca_components}.")

    if control_idx is not None:
        idx = int(control_idx)
        if idx < 0:
            idx += int(X.shape[0])
        if idx < 0 or idx >= int(X.shape[0]):
            raise IndexError(
                f"control_idx={control_idx} is out of bounds for pseudobulk with {X.shape[0]} rows."
            )
        X = np.delete(X, idx, axis=0)
    if X.shape[0] < 2:
        raise ValueError("At least two perturbation pseudobulk rows are required.")
    if float(np.var(X, axis=0, dtype=np.float64).sum()) <= 0.0:
        raise ValueError("Perturbation pseudobulk rows must contain positive variance.")

    n_components = min(int(n_pca_components), int(X.shape[0]), int(X.shape[1]))
    svd_solver = "full" if n_components >= min(X.shape) else "randomized"
    return PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        random_state=random_state,
    ).fit(X)


def estimate_vendi_pseudobulk_sigma_squared(
    ac: AnnData | AnnCollection,
    pca_model: Any,
    layer_key: str,
    control_label: str = "control",
    n_splits: int = 200,
    null_quantile: float = 0.95,
    target_null_similarity: float = 0.95,
    random_state: int = 0,
) -> float:
    """Calibrate pseudobulk outer RBF variance using a precomputed PCA."""
    control_cells, split_size = _prepare_control_null_calibration(
        ac=ac,
        layer_key=layer_key,
        control_label=control_label,
        n_splits=n_splits,
        null_quantile=null_quantile,
        target_null_similarity=target_null_similarity,
    )
    null_distances_squared = np.empty(int(n_splits), dtype=np.float64)
    splits = _iterate_control_splits(control_cells, split_size, n_splits, random_state)
    for split_idx, (first, second) in enumerate(splits):
        first_mean = first.mean(axis=0)
        second_mean = second.mean(axis=0)
        transformed = pca_model.transform(np.vstack([first_mean, second_mean]))
        null_distances_squared[split_idx] = float(np.sum((transformed[0] - transformed[1]) ** 2))

    return _calibrate_outer_sigma_squared(
        null_distances_squared,
        null_quantile,
        target_null_similarity,
    )


def vendi_score_pseudobulk(
    pseudobulk: np.ndarray,  # Pseudobulk matrix of shape (n_perturbations + 1, n_features) or (n_perturbations, n_features) where the first row is control
    control_idx: int | None = None,  # Index of the control row in the pseudobulk matrix
    *,
    pca_model: Any | None,
    outer_sigma_squared: float | None,
) -> float:
    """
    Compute perturbation-level Vendi score using RBF distances.

    Procedure:
     1) Transform perturbation pseudobulks with a PCA fitted on observed
         perturbation pseudobulks: z_i = PCA(x_i).
     2) Build pairwise similarities with the fixed observed-reference variance:
         K_ij = exp(-||z_i-z_j||_2^2 / (2 * sigma^2)).
    3) Normalize matrix as A = K / n_perturbations.
    4) Eigendecompose A to obtain eigenvalues.
    5) Compute von Neumann entropy H(A) = -sum_i lambda_i log(lambda_i), with 0log0 = 0.
    6) Return Vendi score VS = exp(H(A)).

    Args:
        pseudobulk: Pseudobulk matrix of shape (n_perturbations + 1, n_features)
            or (n_perturbations, n_features), where it may contain the control
            row when ``control_idx`` is not ``None``.
        control_idx: Index of the control row in the pseudobulk matrix, or
            ``None`` if no control row is present.
        pca_model: PCA model fitted on observed perturbation pseudobulks and
            shared between observed and predicted scoring. May be ``None`` only
            when at most one distinct perturbation profile is scored.
        outer_sigma_squared: Fixed outer RBF variance calibrated from observed
            same-distribution pseudobulk splits and shared between observed and
            predicted scoring. May be ``None`` only when at most one distinct
            perturbation profile is scored.

    Returns:
        Vendi score computed from the pseudobulk similarity spectrum.
    """
    X = np.asarray(pseudobulk, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"pseudobulk must be a 2D array of shape (n_samples, n_features). Got shape={X.shape}."
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        return float("nan")
    if not np.all(np.isfinite(X)):
        raise ValueError("pseudobulk contains non-finite values (NaN or inf).")

    # Remove the control row when it is explicitly present.
    if control_idx is not None:
        n_rows = int(X.shape[0])
        idx = int(control_idx)
        if idx < 0:
            idx += n_rows
        if idx < 0 or idx >= n_rows:
            raise IndexError(
                f"control_idx={control_idx} is out of bounds for pseudobulk with {n_rows} rows."
            )
        if n_rows <= 1:
            return float("nan")
        row_mask = np.ones(n_rows, dtype=bool)
        row_mask[idx] = False
        X = X[row_mask, :]  # type: ignore[reportConstantRedefinition]

    n_perturbations = int(X.shape[0])
    if n_perturbations == 0:
        return float("nan")
    if n_perturbations == 1:
        return 1.0

    # Constant pseudobulk rows imply zero pairwise distances (all profiles identical).
    # In this case, sklearn PCA emits a divide-by-zero warning when computing
    # explained_variance_ratio_, and the resulting Vendi score should be exactly 1.
    total_var = float(np.var(X, axis=0, dtype=np.float64).sum())
    if total_var <= 0.0:
        return 1.0
    if pca_model is None:
        raise ValueError("pca_model is required when scoring multiple distinct perturbations.")
    if outer_sigma_squared is None or (
        not np.isfinite(outer_sigma_squared) or outer_sigma_squared <= 0.0
    ):
        raise ValueError(
            "outer_sigma_squared must be finite and strictly positive when scoring "
            + f"multiple distinct perturbations. Got {outer_sigma_squared}."
        )

    Z = pca_model.transform(X).astype(np.float64, copy=False)

    dist_sq = pairwise_squared_distances(Z)

    # Similarity matrix from pairwise distances.
    K = np.exp(-dist_sq / (2.0 * outer_sigma_squared), dtype=np.float64)
    np.fill_diagonal(K, 1.0)

    # Normalize and compute Vendi score from spectrum.
    A = K / float(n_perturbations)
    spectrum = np.linalg.eigvalsh(A)
    return _vendi_from_spectrum(spectrum)


def estimate_vendi_outer_sigma_squared(
    ac: AnnData | AnnCollection,
    gamma: float,
    pca_model: Any,
    layer_key: str,
    control_label: str = "control",
    n_splits: int = 200,
    null_quantile: float = 0.95,
    target_null_similarity: float = 0.95,
    random_state: int = 0,
) -> float:
    """Calibrate the outer RBF variance from disjoint observed-control splits."""
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and strictly positive. Got {gamma}.")
    control_cells, split_size = _prepare_control_null_calibration(
        ac=ac,
        layer_key=layer_key,
        control_label=control_label,
        n_splits=n_splits,
        null_quantile=null_quantile,
        target_null_similarity=target_null_similarity,
    )
    control_cells = pca_model.transform(control_cells).astype(np.float64, copy=False)

    null_mmd2 = np.empty(int(n_splits), dtype=np.float64)
    splits = _iterate_control_splits(control_cells, split_size, n_splits, random_state)
    for split_idx, (first, second) in enumerate(splits):
        null_mmd2[split_idx] = biased_mmd2_from_kernel_means(
            kernel_mean(first, kernel="rbf", gamma=gamma),
            kernel_mean(second, kernel="rbf", gamma=gamma),
            kernel_mean(first, second, kernel="rbf", gamma=gamma),
        )

    return _calibrate_outer_sigma_squared(
        null_mmd2,
        null_quantile,
        target_null_similarity,
    )


def vendi_score(
    ac: AnnData | AnnCollection,
    ac_batch_size: int = 1024,
    n_pca_components: int = 50,
    sample_size: int = 2000,
    random_state: int = 0,
    layer_key: str | None = None,
    control_label: str | int | float | None = -1,
    gamma: float | None = None,
    pca_model: Any | None = None,
    *,
    outer_sigma_squared: float | None,
) -> float:
    """
    Compute perturbation-level Vendi score using RBF-MMD distances.

    Procedure:
    1) Choose a feature space:
       use a provided ``pca_model`` when available; otherwise fit PCA only when
       no explicit ``gamma`` is supplied.
    2) If ``gamma`` is not provided, estimate it with the same observed-cell
       heuristic used by ``distribution_distance``.
    3) Build pairwise perturbation distance matrix via biased MMD^2 (RBF kernel).
    4) Convert distances to similarities K_ij = exp(-MMD^2_ij / (2 * sigma^2)).
    5) Normalize matrix as A = K / n_perturbations.
    6) Eigendecompose A to obtain eigenvalues.
    7) Compute von Neumann entropy H(A) = -sum_i lambda_i log(lambda_i), with 0log0 = 0.
    8) Return Vendi score VS = exp(H(A)).

    Args:
        ac: AnnData or AnnCollection containing perturbation-labeled cells.
        ac_batch_size: Number of cells per iteration batch.
        n_pca_components: Number of PCA components before MMD computation.
        sample_size: Number of rows sampled for kernel bandwidth estimation.
        random_state: Random seed used for reproducible sampling.
        layer_key: Optional layer key to use instead of ``.X``.
        control_label: Label in ``.obs["perturbation"]`` identifying control
            cells. If ``None``, no control is excluded.
        gamma: Optional precomputed RBF gamma used inside perturbation-level
            MMD calculations. When provided, vendi_score skips internal gamma
            estimation.
        pca_model: Optional precomputed PCA model defining the feature space for
            MMD calculations. When provided, vendi_score skips internal PCA
            fitting and uses this model for batch transforms.
        outer_sigma_squared: Fixed outer RBF variance used to convert
            perturbation-level MMD^2 distances into similarities. May be
            ``None`` only when zero or one perturbation is scored.

    Returns:
        Vendi score computed from perturbation-level similarities.
    """
    if not obs_has_key(getattr(ac, "obs", None), "perturbation"):
        raise KeyError("ac.obs must contain a 'perturbation' column.")

    obs_pert = np.asarray(ac.obs["perturbation"])  # type: ignore[reportUnknownArgumentType]
    if control_label is None:
        perturbation_ids = np.unique(obs_pert)
    else:
        perturbation_ids = np.unique(obs_pert[obs_pert != control_label])
    try:
        perturbation_ids = np.asarray(sorted(perturbation_ids.tolist()), dtype=object)
    except TypeError:
        perturbation_ids = np.asarray(
            sorted(perturbation_ids.tolist(), key=lambda x: str(x)),
            dtype=object,
        )
    n_perturbations = perturbation_ids.size
    if n_perturbations == 0:
        return float("nan")
    if n_perturbations == 1:
        return 1.0
    if outer_sigma_squared is None or (
        not np.isfinite(outer_sigma_squared) or outer_sigma_squared <= 0.0
    ):
        raise ValueError(
            "outer_sigma_squared must be finite and strictly positive when scoring "
            + f"multiple perturbations. Got {outer_sigma_squared}."
        )

    # Step 1 and Step 2: fit and apply IncrementalPCA according to control_label behavior.
    if pca_model is None and gamma is None:
        if control_label is None:
            pca_model = fit_incremental_pca_all_cells(
                data_obj=ac,
                layer_key=layer_key,
                n_pca_components=n_pca_components,
                batch_size=ac_batch_size,
                data_name="ac",
            )
        else:
            pca_model = fit_control_incremental_pca(
                data_obj=ac,
                layer_key=layer_key,
                control_label=control_label,
                n_pca_components=n_pca_components,
                batch_size=ac_batch_size,
                obs_key="perturbation",
                data_name="ac",
            )

    if gamma is None:
        gamma = estimate_mmd_gamma(
            obs=ac,
            layer_obs=layer_key,
            control_label=cast(Any, control_label),
            seed=random_state,
            gamma_estimation_max_samples=sample_size,
            pca_model=pca_model,
        )

    # Collect one embedding matrix per perturbation ID.
    pert_embeddings_chunks: dict[object, list[np.ndarray]] = {
        ptb: [] for ptb in perturbation_ids.tolist()
    }
    for batch_matrix, batch_labels in iterate_batches(
        data_obj=ac,
        layer_key=layer_key,
        batch_size=ac_batch_size,
        obs_key="perturbation",
    ):
        if batch_labels is None:
            continue
        if control_label is None:
            pert_mask = np.ones(batch_labels.shape[0], dtype=bool)
        else:
            pert_mask = batch_labels != control_label
        if not np.any(pert_mask):
            continue

        batch_pert = np.asarray(batch_matrix[pert_mask, :], dtype=np.float64)
        batch_labels_pert = batch_labels[pert_mask]

        if pca_model is None:
            z_batch = batch_pert
        else:
            z_batch = pca_model.transform(batch_pert).astype(np.float64, copy=False)
        for ptb in np.unique(batch_labels_pert):
            group_mask = batch_labels_pert == ptb
            pert_embeddings_chunks[ptb].append(z_batch[group_mask])

    grouped_embeddings: list[np.ndarray] = []
    for ptb in perturbation_ids.tolist():
        chunks = pert_embeddings_chunks[ptb]
        if not chunks:
            raise ValueError(f"No cells found for perturbation id {ptb!r}.")
        grouped_embeddings.append(np.concatenate(chunks, axis=0))

    # Step 3: compute pairwise biased MMD^2 between perturbation groups.
    mmd2 = np.zeros((n_perturbations, n_perturbations), dtype=np.float64)
    self_kernel_means = np.empty(n_perturbations, dtype=np.float64)
    for i in range(n_perturbations):
        self_kernel_means[i] = kernel_mean(
            grouped_embeddings[i],
            kernel="rbf",
            gamma=gamma,
        )

    for i in range(n_perturbations):
        for j in range(i + 1, n_perturbations):
            cross_mean = kernel_mean(
                grouped_embeddings[i],
                grouped_embeddings[j],
                kernel="rbf",
                gamma=gamma,
            )
            mmd_ij = biased_mmd2_from_kernel_means(
                self_kernel_means[i],
                self_kernel_means[j],
                cross_mean,
            )
            mmd2[i, j] = mmd_ij
            mmd2[j, i] = mmd_ij

    # Step 4: convert MMD^2 distances using the caller-provided outer variance.
    K = np.exp(-mmd2 / (2.0 * outer_sigma_squared), dtype=np.float64)
    np.fill_diagonal(K, 1.0)

    # Normalize and compute Vendi score from spectrum.
    A = K / float(n_perturbations)
    spectrum = np.linalg.eigvalsh(A)
    return _vendi_from_spectrum(spectrum)
