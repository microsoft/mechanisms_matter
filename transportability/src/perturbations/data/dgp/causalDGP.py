"""Synthetic causal DGP utilities for perturbation-based scRNA-seq simulation."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse

from ...analyses.plot_utils import apply_paper_plot_style
from .util import build_obs_block, build_synthetic_adata, sample_nb_counts, sample_unif_pm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
_DEFAULT_MAX_PARALLEL_CHAINS = 64


def softplus_beta(x: np.ndarray, beta: float) -> np.ndarray:
    """
    Stable softplus with temperature/beta: (1/beta) * log(1 + exp(beta * x)).

    As beta -> inf, this approaches ReLU. As beta -> 0, it approaches identity.
    """
    z = beta * x
    return np.logaddexp(0.0, z) / beta


def _build_matrix_erdos_renyi(
    G: int,
    rng: np.random.Generator,
    expected_edges_per_gene: int = 10,
) -> sparse.csr_matrix:
    """
    Build sparse with ~expected_edges_per_gene nonzeros per row (target gene).

    i.e., ~10 regulators per gene.

    A_{i,j} != 0 means gene j directly influences gene i in the linear drift.
    """
    edges_per_gene = min(expected_edges_per_gene, G - 1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(G):
        # choose regulators j != i
        # (for large G, this is fast enough and avoids dense masks)
        regs = rng.choice(G - 1, size=edges_per_gene, replace=False)
        regs = regs + (regs >= i)  # shift up to skip i
        w = sample_unif_pm(rng, 1.0, 3.0, size=edges_per_gene)
        rows.extend([i] * edges_per_gene)
        cols.extend(regs.tolist())
        data.extend(w.tolist())

    A = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(G, G),
    )
    return A


def _build_matrix_power_law(
    G: int,
    rng: np.random.Generator,
    m: int = 10,
    strength: float = 2.0,
    flip_prob: float = 0.1,
) -> sparse.csr_matrix:
    """
    Barabasi-Albert-style preferential attachment with exponent 'strength' (degree**strength).

    with approximately m links per gene on average (not a hard fixed count per new node).
    Edge direction is flipped with probability flip_prob to create feedback loops.

    This is O(G^2) in the naive implementation due to computing probabilities each step.
    For G=10_000 it can still be OK on HPC, but Erdos-Renyi is much faster.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Start with a small seed set. We do not require m0 >= m because m is treated
    # as an average number of links, and per-node links are sampled stochastically.
    m0 = min(12, G)
    degrees = np.zeros(G, dtype=np.float64)

    # Seed: connect nodes [0..m0-1] in a simple chain to initialize degrees
    for i in range(m0 - 1):
        j = i + 1
        degrees[i] += 1
        degrees[j] += 1

        # Default direction older -> newer
        src, dst = i, j
        if rng.random() < flip_prob:
            src, dst = dst, src

        rows.append(dst)
        cols.append(src)
        data.append(float(sample_unif_pm(rng, 1.0, 3.0, size=()).item()))

    # Grow graph
    for new in range(m0, G):
        # attachment probs proportional to degree**strength (plus tiny epsilon)
        deg = degrees[:new]
        w = (deg + 1e-9) ** strength
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            p = np.full(new, 1.0 / new, dtype=np.float64)
        else:
            p = w / w_sum

        # "10 links per gene" is interpreted as an average rather than fixed links.
        # Sample links per new node from a Poisson distribution centered at m.
        k_links = int(rng.poisson(lam=max(float(m), 0.0)))
        k_links = max(1, min(k_links, new))
        targets = rng.choice(new, size=k_links, replace=False, p=p)

        for old in targets:
            # Default orientation: old -> new (hubs become regulators)
            src, dst = int(old), int(new)
            if rng.random() < flip_prob:
                src, dst = dst, src

            rows.append(dst)
            cols.append(src)
            data.append(float(sample_unif_pm(rng, 1.0, 3.0, size=()).item()))

            degrees[old] += 1
            degrees[new] += 1

    A = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(G, G),
    )
    return A


def _shift_causal_matrix(
    A: sparse.csr_matrix, target_max_real_eig: float = -0.5
) -> sparse.csr_matrix:
    """
    Shift the causal matrix by a constant diagonal so that max real-part eigenvalue <= target_max_real_eig.

    We try ARPACK eigs(which='LR') on the non-symmetric A. If it fails, we fall back to using
    the symmetric part (A + A.T)/2 which provides an upper bound on spectral abscissa.

    ARPACK draws a random starting vector when ``v0`` is not supplied, so at the loose
    ``tol`` used here the returned eigenvalue varies between otherwise identical calls.
    That jitter propagates into the diagonal shift below and makes the whole generator
    irreproducible even at a fixed ``seed``, because ARPACK's randomness lives in its own
    internal state rather than in the caller's seeded ``Generator``. Pin ``v0`` so the
    estimate is a deterministic function of ``A``.
    """
    G = A.shape[0]
    v0 = np.ones(G, dtype=np.float64)
    try:
        # largest real part eigenvalue estimate
        vals = spla.eigs(
            A, k=1, which="LR", return_eigenvectors=False, tol=1e-2, maxiter=2000, v0=v0
        )
        s_est = float(np.max(np.real(vals)))
    except Exception:
        # fallback: use symmetric part upper bound
        As = (A + A.T).multiply(0.5)
        vals = spla.eigsh(
            As, k=1, which="LA", return_eigenvectors=False, tol=1e-2, maxiter=2000, v0=v0
        )
        s_est = float(vals[0])

    A_stable = A - (s_est - target_max_real_eig) * sparse.eye(G, format="csr", dtype=np.float32)
    return A_stable


def _build_base_matrix(G: int, rng: np.random.Generator, mask_method: str) -> sparse.csr_matrix:
    method = mask_method.strip().lower().replace("_", "-")
    builders: dict[str, Callable[..., sparse.csr_matrix]] = {
        "power-law": _build_matrix_power_law,
        "erdos-renyi": _build_matrix_erdos_renyi,
    }
    if method not in builders:
        raise ValueError(f"Unknown mask_method: {mask_method}")
    return builders[method](G=G, rng=rng)


def _build_bc_vectors(bias: np.ndarray, c_q: np.ndarray) -> list[np.ndarray]:
    out = [bias.copy()]
    out.extend(bias + c_q[p] for p in range(c_q.shape[0]))
    return out


def _swap_matrix(
    A: sparse.csr_matrix,
    rng: np.random.Generator,
    swap_fraction: float = 0.2,
) -> sparse.csr_matrix:
    """
    Create a row-wise perturbed copy of A by swapping a fraction of nonzero entries.

    with zero locations in the same row.

    For each row:
      - choose floor(nnz_row * swap_fraction) existing edges
      - move those edge weights to currently-zero columns (without replacement)
      - keep row nnz unchanged and preserve weight distribution
    """
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    A = A.astype(np.float32, copy=True)
    G = A.shape[1]

    new_indices_rows: list[np.ndarray] = []
    new_data_rows: list[np.ndarray] = []
    new_indptr = np.zeros(A.shape[0] + 1, dtype=np.int64)

    for i in range(A.shape[0]):
        start = A.indptr[i]
        end = A.indptr[i + 1]
        row_cols = A.indices[start:end].astype(np.int64, copy=True)
        row_vals = A.data[start:end].astype(np.float32, copy=True)
        row_nnz = row_cols.size

        if row_nnz == 0:
            new_indices_rows.append(np.empty(0, dtype=np.int32))
            new_data_rows.append(np.empty(0, dtype=np.float32))
            new_indptr[i + 1] = new_indptr[i]
            continue

        n_swap = math.floor(row_nnz * float(swap_fraction))
        if n_swap <= 0:
            row_cols_out = row_cols.astype(np.int32, copy=False)
            row_vals_out = row_vals
        else:
            swap_pos = rng.choice(row_nnz, size=n_swap, replace=False)
            keep_mask = np.ones(row_nnz, dtype=bool)
            keep_mask[swap_pos] = False

            keep_cols = row_cols[keep_mask]
            keep_vals = row_vals[keep_mask]
            moved_vals = row_vals[swap_pos]

            occupied = np.zeros(G, dtype=bool)
            occupied[row_cols] = True
            # keep no-self-edge behavior when possible
            if i < G:
                occupied[i] = True

            candidate_zero_cols = np.flatnonzero(~occupied)
            if candidate_zero_cols.size < n_swap:
                # fallback: allow self-edge if needed
                if i < G:
                    occupied[i] = row_cols.__contains__(i)
                candidate_zero_cols = np.flatnonzero(~occupied)

            if candidate_zero_cols.size < n_swap:
                n_swap = candidate_zero_cols.size
                if n_swap == 0:
                    row_cols_out = row_cols.astype(np.int32, copy=False)
                    row_vals_out = row_vals
                    new_indices_rows.append(row_cols_out)
                    new_data_rows.append(row_vals_out)
                    new_indptr[i + 1] = new_indptr[i] + row_cols_out.size
                    continue
                moved_vals = moved_vals[:n_swap]

            new_cols = rng.choice(candidate_zero_cols, size=n_swap, replace=False)
            row_cols_out = np.concatenate([keep_cols, new_cols]).astype(np.int32, copy=False)
            row_vals_out = np.concatenate([keep_vals, moved_vals]).astype(np.float32, copy=False)

            order = np.argsort(row_cols_out, kind="mergesort")
            row_cols_out = row_cols_out[order]
            row_vals_out = row_vals_out[order]

        new_indices_rows.append(row_cols_out)
        new_data_rows.append(row_vals_out)
        new_indptr[i + 1] = new_indptr[i] + row_cols_out.size

    if new_indptr[-1] == 0:
        return sparse.csr_matrix(A.shape, dtype=np.float32)

    new_indices = np.concatenate(new_indices_rows).astype(np.int32, copy=False)
    new_data = np.concatenate(new_data_rows).astype(np.float32, copy=False)
    return sparse.csr_matrix((new_data, new_indices, new_indptr), shape=A.shape, dtype=np.float32)


def _draw_condition_states(
    *,
    A_list: list[sparse.csr_matrix],
    bc_list: list[list[np.ndarray]],
    bc_index: int,
    cell_line_batch: np.ndarray,
    rng: np.random.Generator,
    dt: float,
    burn_in_steps: int,
    thinning_steps: int,
    max_parallel_chains: int = _DEFAULT_MAX_PARALLEL_CHAINS,
) -> np.ndarray:
    """Draw latent states for one condition using fresh condition-specific samplers."""
    if max_parallel_chains < 1:
        raise ValueError(f"max_parallel_chains must be >= 1, got {max_parallel_chains}.")

    n_cells = int(cell_line_batch.size)
    G = int(A_list[0].shape[0])
    x_batch = np.empty((n_cells, G), dtype=np.float32)

    for cell_line_id in (0, 1):
        idx = np.flatnonzero(cell_line_batch == cell_line_id)
        if idx.size == 0:
            continue

        sampler = EMSampler(
            A=A_list[cell_line_id],
            bc=bc_list[cell_line_id][bc_index],
            rng=rng,
            dt=dt,
            sigma=math.sqrt(2.0),
            burn_in_steps=burn_in_steps,
            thinning_steps=thinning_steps,
            chains=min(int(idx.size), max_parallel_chains),
            dtype=np.float32,
        )
        samples = _draw_samples(sampler, int(idx.size))
        x_batch[idx] = samples

    return x_batch


def _draw_samples(sampler: EMSampler, n_samples: int) -> np.ndarray:
    """Draw any number of samples from a capped-width sampler."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}.")
    if n_samples == 0:
        return np.empty((0, sampler.A.shape[0]), dtype=sampler.dtype)

    out = np.empty((n_samples, sampler.A.shape[0]), dtype=sampler.dtype)
    filled = 0
    while filled < n_samples:
        chunk_size = min(sampler.chains, n_samples - filled)
        out[filled : filled + chunk_size] = sampler.draw(chunk_size)
        filled += chunk_size
    return out


def _solve_stationary_means(
    A: sparse.csr_matrix,
    bc_vectors: list[np.ndarray],
) -> np.ndarray:
    """
    Solve stationary means for all conditions of one cell line at once.

    For dx = (A x + bc) dt + sqrt(2) dW with stable A, the stationary mean solves:
        A mu + bc = 0  =>  mu = -A^{-1} bc
    """
    rhs = np.column_stack([np.asarray(bc, dtype=np.float32) for bc in bc_vectors])
    lu = spla.splu(A.tocsc())
    means = -lu.solve(rhs)
    means_arr = np.asarray(means, dtype=np.float32)
    if means_arr.ndim == 1:
        means_arr = means_arr[:, None]
    return np.ascontiguousarray(means_arr.T)


@dataclass
class EMSampler:
    """
    Parallel Euler-Maruyama sampler for.

        dx = (A x + bc) dt + sigma dW

    We maintain 'chains' parallel chains of dimension G and stream out samples.
    """

    A: sparse.csr_matrix  # (G,G) sparse
    bc: np.ndarray  # (G,) float32, bc = b + c_q
    rng: np.random.Generator
    dt: float = 1e-3
    sigma: float = math.sqrt(2.0)
    burn_in_steps: int = 200
    thinning_steps: int = 20
    chains: int = 64
    dtype: type = np.float32

    def __post_init__(self):
        """Initialize state vectors and run burn-in for all chains."""
        G = self.A.shape[0]
        self.bc = np.asarray(self.bc, dtype=self.dtype)
        self.x = self.rng.normal(0.0, 1.0, size=(self.chains, G)).astype(self.dtype, copy=False)

        # Burn-in once per condition to reduce dependence on initialization
        self._step(self.burn_in_steps)

    def _drift(self, x: np.ndarray) -> np.ndarray:
        # drift = (A @ x^T)^T + bc
        lin = (self.A @ x.T).T  # (chains,G)
        lin += self.bc[None, :]
        return lin

    def _step(self, n_steps: int):
        if n_steps <= 0:
            return
        dt = float(self.dt)
        noise_scale = float(self.sigma) * math.sqrt(dt)

        for _ in range(n_steps):
            drift = self._drift(self.x)
            noise = self.rng.normal(0.0, 1.0, size=self.x.shape).astype(self.dtype, copy=False)
            self.x = self.x + dt * drift + noise_scale * noise

    def draw(self, n_samples: int) -> np.ndarray:
        """Draw n_samples latent states as an array of shape (n_samples, G)."""
        if n_samples < 0:
            raise ValueError(f"n_samples must be non-negative, got {n_samples}.")
        if n_samples > self.chains:
            raise ValueError(
                f"n_samples ({n_samples}) cannot exceed configured chains ({self.chains})."
            )
        out = self.x[:n_samples].copy()
        self._step(self.thinning_steps)
        return out


def causalDGP(
    G: int,
    N0: int,
    Nk: int,
    P: int,
    mu_l: float,
    all_theta: np.ndarray,
    gene_names: np.ndarray,
    control_label: str = "control",
    mask_method: str = "Erdos-Renyi",  # NOTE: "Erdos-Renyi" is much more easier to generate a distinguishable patterns than "power-law"
    diversity_type: str = "A",
    swap_fraction: float = 0.8,
    seed: int | None = None,
    normalize: bool = True,
    normalized_layer_key: str = "normalized_log1p",
    visualize: bool = False,
    verbose: bool = False,
    output_dir: str | None = None,
) -> tuple[ad.AnnData, list[np.ndarray]]:
    """
    End-to-end synthetic scRNA-seq generator.

    Latent dynamics (Euler-Maruyama only):
        dx = (A x + b + c_q) dt + sqrt(2) dW
    Observation model (ZIP):
        y_g = 0 w.p. pi_g
        else y_g ~ Poisson( eta_g * softplus(x_g) )

    Args:
        G: Number of genes.
        N0: Number of control cells.
        Nk: Number of perturbed cells per perturbation.
        P: Number of perturbations.
        mu_l: Mean of log library size (log-normal location parameter).
        all_theta: Dispersion parameters used for negative binomial sampling.
        gene_names: Gene name array used to label sampled genes.
        control_label: Label used for the control condition.
        mask_method: Base causal matrix generator (Erdos-Renyi or power-law).
        diversity_type: Diversity mode across cell lines: "A", "b", "both" or "none".
        swap_fraction: Row-wise fraction of edges rewired for `A_alter`.
        seed: Optional RNG seed for reproducibility.
        normalize: Whether to add a normalized/log1p layer to the output.
        normalized_layer_key: Layer key used for normalized/log1p values.
        visualize: Whether to save diagnostic matrix visualizations.
        verbose: Whether to print detailed information during execution.
        output_dir: Directory for visualization files when `visualize=True`.

    Returns:
      - adata: AnnData storing raw counts in `.X` and normalized/log1p values in the requested layer
      - all_affected_masks: list[list[np.ndarray]], two lists of boolean masks (one per perturbation)
      for A and A_alter that indicate which genes are affected by each perturbation.
    """
    rng = np.random.default_rng(42 if seed is None else seed)

    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if N0 < 0 or Nk < 0 or P < 0:
        raise ValueError(f"N0, Nk, and P must be non-negative, got N0={N0}, Nk={Nk}, P={P}")
    if not (0.0 <= swap_fraction <= 1.0):
        raise ValueError(f"swap_fraction must be in [0, 1], got {swap_fraction}")

    assert isinstance(all_theta, np.ndarray), "all_theta must be a numpy array"
    # Assert that G is not larger than the provided arrays
    assert len(all_theta) >= G, (
        f"G parameter ({G}) cannot be larger than the length of provided arrays ({len(all_theta)})"
    )
    if gene_names is not None:
        gene_names_arr = np.asarray(gene_names, dtype=str)
        assert len(gene_names_arr) >= G, (
            f"gene_names must have at least G entries. Got len(gene_names)={len(gene_names_arr)}, G={G}"
        )
        assert np.unique(gene_names_arr).size >= G, (
            "gene_names must contain at least G unique names so sampled genes stay uniquely identifiable."
        )
    else:
        gene_names_arr = np.asarray([f"gene_{i}" for i in range(len(all_theta))], dtype=str)

    # Sample G elements from all_theta
    indices = rng.choice(len(all_theta), size=G, replace=False)
    local_all_theta = all_theta[indices]  # Use the all-cells theta
    local_gene_names = gene_names_arr[indices]

    # Build shift function f_q(x) = Ax + b + c_q.
    A = _build_base_matrix(G=G, rng=rng, mask_method=mask_method)

    # Create a perturbed version of A by swapping X% (swap_fraction) of each row's nonzero edges
    # into zero positions. This preserves row sparsity and weight distribution.
    # NOTE: tweak target_max_real_eig
    diversity_type = diversity_type.strip().lower()
    if diversity_type in ("a", "both"):
        A_alter = _swap_matrix(A=A, rng=rng, swap_fraction=swap_fraction)
        A = _shift_causal_matrix(A, target_max_real_eig=-1)
        A_alter = _shift_causal_matrix(A_alter, target_max_real_eig=-1)
    elif diversity_type in ("b", "none"):
        A = _shift_causal_matrix(A, target_max_real_eig=-1)
        A_alter = A.copy()
    else:
        raise ValueError(
            f"Invalid diversity_type: {diversity_type}. Must be one of 'A', 'b', 'both', or 'none'."
        )

    b_base = rng.uniform(-3.0, 3.0, size=G).astype(np.float32)
    if diversity_type in ("b", "both"):
        b_base_alt = rng.uniform(-3.0, 3.0, size=G).astype(np.float32)
    elif diversity_type in ("a", "none"):
        b_base_alt = b_base.copy()
    else:
        raise ValueError(
            f"Invalid diversity_type: {diversity_type}. Must be one of 'A', 'b', 'both', or 'none'."
        )

    # Randomly reorder genes in A and b
    perm = rng.permutation(G)
    A = A[:, perm][perm, :]
    A_alter = A_alter[:, perm][perm, :]
    b_base = b_base[perm]
    b_base_alt = b_base_alt[perm]
    local_all_theta = local_all_theta[perm]
    local_gene_names = local_gene_names[perm]

    A_list = [A, A_alter]
    b_list = [b_base, b_base_alt]

    # sample perturbations c_q
    # Control is q=-1 with c=0. Perturbations are q=0..P-1.
    assert P <= G, (
        f"The possible number of perturbation targets exceeds total genes. Reduce P. Got P={P}, G={G}"
    )
    targets = rng.choice(G, size=P, replace=False).astype(np.int64)
    targets.sort()  # for better visualization in the color-map
    shifts = sample_unif_pm(rng, 2 * np.log(G), 6 * np.log(G), size=(P)).astype(np.float32)
    c_q = np.zeros((P, G), dtype=np.float32)
    np.add.at(c_q, (np.arange(P), targets), shifts)

    if visualize:
        if output_dir is None:
            raise ValueError("output_dir must be provided when visualize=True.")
        apply_paper_plot_style()
        # Build stacked matrices [A^T; b] and [A_alter^T; b_alter],
        # each with shape (G + 1, G).
        Ab = np.vstack([A.toarray().T.astype(np.float32, copy=False), b_list[0][None, :]])
        Ab_alter = np.vstack(
            [A_alter.toarray().T.astype(np.float32, copy=False), b_list[1][None, :]]
        )

        # Black around 0, red for negative, blue for positive.
        # With vmin/vmax set to [-3, 3], 0 map to 0.5 on the color axis.
        cmap_1 = LinearSegmentedColormap.from_list(
            "red_black_blue",
            [
                (0.00, "#fdabab"),  # strong negative -> red
                (0.50, "#000000"),  # 0 -> black
                (1.00, "#9baffe"),  # strong positive -> blue
            ],
            N=256,
        )
        vmin, vmax = -3.0, 3.0

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), constrained_layout=True)
        axes[0].imshow(
            Ab, cmap=cmap_1, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
        )
        im1 = axes[1].imshow(
            Ab_alter, cmap=cmap_1, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
        )

        axes[0].set_title(r"$[A_{C1}^\top; B_{C1}]$")
        axes[1].set_title(r"$[A_{C2}^\top; B_{C2}]$")
        for ax in axes:
            ax.set_xlabel("Genes (downstream)")
            ax.set_ylabel("Bias + Genes (upstream)")
            ax.set_yticks([0, G])
            ax.set_yticklabels(["0", "b"])
            ax.tick_params(axis="y", length=0)
            ax.axhline(G - 0.5, color="white", linewidth=1.2, alpha=0.85)
            for spine in ax.spines.values():
                spine.set_visible(False)

        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("Value")
        cbar.outline.set_visible(False)

        out_path = Path(output_dir) / "causal_effect_visualization.svg"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved causal effect heatmap visualization to: {out_path}")

        cmap_2 = LinearSegmentedColormap.from_list(
            "orange_white_green",
            [
                (0.00, "#faf9d1"),
                (0.50, "#000000"),
                (1.00, "#e1e2fd"),
            ],
            N=256,
        )
        cq_fig, cq_ax = plt.subplots(
            figsize=(12.8, max(2.4, 0.42 * P)),
            constrained_layout=True,
        )
        cq_im = cq_ax.imshow(
            c_q,
            cmap=cmap_2,
            vmin=-15.0,
            vmax=15.0,
            aspect="auto",
            interpolation="nearest",
        )
        cq_ax.set_title(r"Shifts $\left[c_{q_1}, \dots, c_{q_k}\right]^\top$")
        cq_ax.set_xlabel("Gene index")
        cq_ax.set_ylabel("Perturbations")
        for spine in cq_ax.spines.values():
            spine.set_visible(False)
        cq_cbar = cq_fig.colorbar(cq_im, ax=cq_ax, shrink=0.85, pad=0.02)
        cq_cbar.set_label("Shift value")
        cq_cbar.outline.set_visible(False)

        cq_out_path = Path(output_dir) / "perturbation_shift.svg"
        cq_fig.savefig(cq_out_path)
        plt.close(cq_fig)
        print(f"Saved perturbation shift heatmap visualization to: {cq_out_path}")

    # Store bc vectors for each cell line as (b_type + c_q).
    # Index 0 corresponds to control bc (q=-1), then perturbations q=0..P-1.
    bc_list: list[list[np.ndarray]] = [_build_bc_vectors(bias=b, c_q=c_q) for b in b_list]
    stationary_means = [
        _solve_stationary_means(A=A_matrix, bc_vectors=bc_vectors)
        for A_matrix, bc_vectors in zip(A_list, bc_list, strict=True)
    ]

    # Perturbation effects are the difference from the control stationary mean.
    all_affected_masks = [
        np.abs(stationary_means[0][p + 1] - stationary_means[0][0]) > 1e-6 for p in range(P)
    ]
    all_affected_masks_alter = [
        np.abs(stationary_means[1][p + 1] - stationary_means[1][0]) > 1e-6 for p in range(P)
    ]

    perturbation_names = np.asarray(
        [str(local_gene_names[int(np.flatnonzero(c_q[p])[0])]) for p in range(P)],
        dtype=str,
    )

    var = pd.DataFrame(index=pd.Index(local_gene_names, name="gene"))
    counts_blocks: list[sparse.csr_matrix] = []
    obs_blocks: list[pd.DataFrame] = []

    # EM sampling parameters
    dt = 1e-3
    burn_in_steps = 12000
    thinning_steps = 10

    # condition order: control (-1), then perturbations 0..P-1
    conditions = [(-1, N0, 0)] + [
        (p, Nk, p + 1) for p in range(P)
    ]  # (perturbation_id, n_cells, bc_index)
    sampled_control_means: dict[int, np.ndarray] = {}

    for perturbation_id, n_cells, bc_index in conditions:
        perturbation_name = (
            control_label if perturbation_id < 0 else str(perturbation_names[perturbation_id])
        )
        cell_line_batch = rng.binomial(1, 0.5, size=n_cells).astype(np.int32, copy=False)
        x_batch = _draw_condition_states(
            A_list=A_list,
            bc_list=bc_list,
            bc_index=bc_index,
            cell_line_batch=cell_line_batch,
            rng=rng,
            dt=dt,
            burn_in_steps=burn_in_steps,
            thinning_steps=thinning_steps,
        )

        # NOTE: tweaking the beta, lib_size
        mu_batch = softplus_beta(x_batch, beta=3).astype(np.float32, copy=False)
        lib_size = np.ones(n_cells, dtype=np.float32)
        # lib_size = rng.lognormal(
        #     mean=mu_l, sigma=0.1714, size=n_cells
        # )  # 0.1714 from all cells of the Norman19 dataset
        # replace the gene-specific dispersion parameters with a constant across all genes
        # local_all_theta = np.full_like(local_all_theta, fill_value=local_all_theta.mean())
        counts_blocks.append(
            sample_nb_counts(mean=mu_batch, l_c=lib_size, theta=local_all_theta, rng=rng)
        )

        obs_blocks.append(
            build_obs_block(
                n_cells=n_cells,
                perturbation_id=perturbation_id,
                perturbation_name=perturbation_name,
                cell_line=cell_line_batch,
            )
        )

        if verbose:
            for cell_line_id in (0, 1):
                line_idx = cell_line_batch == cell_line_id
                if not np.any(line_idx):
                    continue
                sampled_mean = np.asarray(x_batch[line_idx].mean(axis=0), dtype=np.float32)
                mean_error = np.asarray(
                    sampled_mean - stationary_means[cell_line_id][bc_index], dtype=np.float32
                )
                rmse = float(np.sqrt(np.mean(mean_error * mean_error)))
                max_abs = float(np.max(np.abs(mean_error)))
                print(
                    f"[EM check] perturbation={perturbation_name} cell_line={cell_line_id} "
                    f"rmse={rmse:.4f} max_abs={max_abs:.4f} n={int(np.sum(line_idx))}"
                )
                if perturbation_id < 0:
                    sampled_control_means[cell_line_id] = sampled_mean
                elif cell_line_id in sampled_control_means:
                    sampled_shift = sampled_mean - sampled_control_means[cell_line_id]
                    analytic_shift = (
                        stationary_means[cell_line_id][bc_index] - stationary_means[cell_line_id][0]
                    )
                    print(
                        f"[Shift check] perturbation={perturbation_name} cell_line={cell_line_id} "
                        f"sampled_norm={np.linalg.norm(sampled_shift):.4f} "
                        f"analytic_norm={np.linalg.norm(analytic_shift):.4f}"
                    )

    return build_synthetic_adata(
        counts_blocks=counts_blocks,
        obs_blocks=obs_blocks,
        var=var,
        normalize=normalize,
        normalized_layer_key=normalized_layer_key,
    ), [all_affected_masks, all_affected_masks_alter]
