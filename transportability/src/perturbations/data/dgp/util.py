"""Utility functions for synthetic DGP data generation."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _normalize_and_log1p(matrix, normalize=True, target_sum=1e4):
    """
    Normalize each cell to target_sum and apply log1p.

    Operates on CSR sparse matrices or dense numpy arrays. Returns the same type as input.
    """
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        if normalize:
            lib_sizes = np.asarray(matrix.sum(axis=1)).ravel()
            # Avoid division by zero for cells with zero total counts.
            lib_sizes[lib_sizes == 0.0] = 1.0
            scale = (target_sum / lib_sizes).astype(np.float32, copy=False)
            matrix = matrix.multiply(scale[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
    else:
        matrix = np.asarray(matrix, dtype=np.float32).copy()
        if normalize:
            lib_sizes = matrix.sum(axis=1, dtype=np.float32)
            lib_sizes[lib_sizes == 0.0] = 1.0
            matrix *= (target_sum / lib_sizes)[:, None]
        np.log1p(matrix, out=matrix)

    return matrix


def sample_unif_pm(rng: np.random.Generator, low: float, high: float, size) -> np.ndarray:
    """Uniform magnitude in [low, high] times random sign ±1."""
    mag = rng.uniform(low, high, size=size)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=size)
    return mag * sign


def sample_nb_counts(
    mean, l_c, theta, rng
):  # theta kept as generic parameter name for this utility function
    """
    Generate individual cell profiles from NB distribution.

    Returns an array of shape (len(l_c), G).
    """
    # Ensure mean and theta are numpy arrays for element-wise operations
    mean_arr = np.asarray(mean, dtype=np.float32)
    theta_arr = np.asarray(theta, dtype=np.float32)
    l_c_arr = np.asarray(l_c, dtype=np.float32)

    # Correct mean for library size
    if mean_arr.ndim == 1:
        lib_size_corrected_mean = np.outer(l_c_arr, mean_arr)
    else:
        lib_size_corrected_mean = l_c_arr[:, None] * mean_arr

    # Prevent division by zero or negative p if theta + mean is zero or mean is much larger than theta
    # This can happen if means are very low and theta is also low.
    # Add a small epsilon to the denominator to stabilize.
    # Also ensure p is within (0, 1)
    p_denominator = theta_arr + lib_size_corrected_mean
    p_denominator[p_denominator <= 0] = 1e-9  # Avoid zero or negative denominator

    p = theta_arr / p_denominator
    p = np.clip(p, 1e-9, 1 - 1e-9)  # Ensure p is in a valid range for negative_binomial

    # Negative binomial expects n (number of successes, our theta) to be > 0.
    # And p (probability of success) to be in [0, 1].
    # If theta contains zeros or negatives, np.random.negative_binomial will fail.
    # Assuming theta values are appropriate (positive).

    predicted_counts = rng.negative_binomial(theta_arr, p).astype(np.int32, copy=False)
    return sparse.csr_matrix(predicted_counts, dtype=np.int32)


def build_obs_block(
    n_cells: int,
    perturbation_id: int,
    perturbation_name: str,
    cell_line: int | np.ndarray,
) -> pd.DataFrame:
    """Build an obs block for one synthetic condition."""
    if np.isscalar(cell_line):
        cell_line_arr = np.full(n_cells, int(cell_line), dtype=np.int32)
    else:
        cell_line_arr = np.asarray(cell_line, dtype=np.int32)
        if cell_line_arr.shape != (n_cells,):
            raise ValueError(f"cell_line must have shape ({n_cells},), got {cell_line_arr.shape}.")

    return pd.DataFrame(
        {
            "perturbation": np.full(n_cells, perturbation_name, dtype=object),
            "perturbation_id": np.full(n_cells, perturbation_id, dtype=np.int32),
            "cell_line": cell_line_arr,
        }
    )


def build_synthetic_adata(
    counts_blocks: list[sparse.csr_matrix],
    obs_blocks: list[pd.DataFrame],
    var: pd.DataFrame,
    normalize: bool = True,
    normalized_layer_key: str = "normalized_log1p",
) -> ad.AnnData:
    """
    Build a single in-memory AnnData object from synthetic count and obs blocks.

    The returned AnnData stores raw counts in `.X` and the normalized/log1p matrix
    in `layers[normalized_layer_key]`.
    """
    if not counts_blocks:
        raise ValueError("counts_blocks must contain at least one synthetic block.")
    if len(counts_blocks) != len(obs_blocks):
        raise ValueError(
            f"counts_blocks and obs_blocks must have the same length. "
            f"Got {len(counts_blocks)} and {len(obs_blocks)}."
        )

    counts = sparse.vstack(
        [block.tocsr().astype(np.int32, copy=False) for block in counts_blocks],
        format="csr",
        dtype=np.int32,
    )
    obs = pd.concat(obs_blocks, axis=0, ignore_index=True)
    if counts.shape[0] != len(obs):
        raise ValueError(f"Counts rows ({counts.shape[0]}) do not match obs rows ({len(obs)}).")

    obs.index = [f"cell_{i}" for i in range(counts.shape[0])]
    adata = ad.AnnData(X=counts, obs=obs, var=var.copy())
    adata.layers[normalized_layer_key] = _normalize_and_log1p(
        counts,
        normalize=normalize,
    )
    return adata
