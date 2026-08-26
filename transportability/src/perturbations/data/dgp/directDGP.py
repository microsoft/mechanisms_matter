"""Direct generative model for synthetic perturbation data."""

import anndata as ad
import numpy as np
import pandas as pd

from .util import build_obs_block, build_synthetic_adata, sample_nb_counts


def directDGP(
    G,  # number of genes
    N0,  # number of control cells
    Nk,  # number of perturbed cells per perturbation
    P,  # number of perturbations
    p_effect,  # a threshold for fraction of genes affected per perturbation
    effect_factor,  # effect factor for affected genes, epsilon in the paper
    B,  # global perturbation bias factor, beta in the paper
    mu_l,  # mean of log library size
    all_theta,  # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    control_mu,  # Control mu parameters, size of total number of genes in the real dataset (>= G)
    pert_mu,  # Perturbed mu parameters, size of total number of genes in the real dataset (>= G)
    gene_names: np.ndarray,  # Optional real gene names to assign to the sampled genes
    control_label: str = "control",
    seed: int | None = None,
    normalize: bool = True,  # Whether to normalize before log1p for the persisted layer
    normalized_layer_key: str = "normalized_log1p",  # Layer name for normalized/log1p values
) -> tuple[ad.AnnData, list[np.ndarray]]:
    """
    Generate a synthetic perturbation dataset as an in-memory AnnData object.

    The returned AnnData stores raw counts in `.X` and normalized/log1p values in
    `layers[normalized_layer_key]`.

    Returns:
      - adata: AnnData with synthetic counts and per-cell metadata
      - all_affected_masks: list[np.ndarray], one mask per perturbation
    """
    rng = np.random.default_rng(42 if seed is None else seed)

    # --- Parameter Preparation with assertions ---
    # Assert that inputs are already arrays
    assert isinstance(control_mu, np.ndarray), "control_mu must be a numpy array"
    assert isinstance(pert_mu, np.ndarray), "pert_mu must be a numpy array"
    assert isinstance(all_theta, np.ndarray), "all_theta must be a numpy array"
    # Assert that they have the same length
    assert len(control_mu) == len(all_theta), "control_mu and all_theta must have the same length."
    assert len(control_mu) == len(pert_mu), "control_mu and pert_mu must have the same length."
    # Assert that G is not larger than the provided arrays
    assert len(control_mu) >= G, (
        f"G parameter ({G}) cannot be larger than the length of provided arrays ({len(control_mu)})"
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
        gene_names_arr = np.asarray([f"gene_{i}" for i in range(len(control_mu))], dtype=str)
    # --- End of assertions ---

    # Sample G elements from control_mu, all_theta, and pert_mu to define the local parameters for selected genes
    indices = rng.choice(len(control_mu), size=G, replace=False)
    local_control_mu = control_mu[indices]
    local_all_theta = all_theta[indices]  # Use the all-cells theta
    local_pert_mu = pert_mu[indices]
    local_gene_names = gene_names_arr[indices]

    var = pd.DataFrame(index=pd.Index(local_gene_names, name="gene"))

    all_affected_masks = []
    counts_blocks: list = []
    obs_blocks: list[pd.DataFrame] = []

    # 1. Sample control cells with bias (B, dispersion set to all_theta from all cells, fixed dispersion assumption)
    lib_size_control = rng.lognormal(
        mean=mu_l, sigma=0.1714, size=N0
    )  # 0.1714 from all cells of the Norman19 dataset
    control_counts = sample_nb_counts(
        mean=local_control_mu, l_c=lib_size_control, theta=local_all_theta, rng=rng
    )
    counts_blocks.append(control_counts)
    obs_blocks.append(
        build_obs_block(
            n_cells=N0,
            perturbation_id=-1,
            perturbation_name=control_label,
            cell_line=0,
        )
    )

    # Define global perturbation bias, this is the terms in brackets for eq 2 in the paper
    delta_b = local_pert_mu - local_control_mu
    local_pert_mu_biased = np.clip(local_control_mu + B * delta_b, 0.0, np.inf)

    # 2. For each perturbation generate the cells
    perturbation_names = rng.choice(local_gene_names, size=P, replace=False)
    for perturbation_id in range(P):
        # this is eq 4 in the paper
        affected_mask_loop = rng.random(G) < p_effect
        all_affected_masks.append(affected_mask_loop)

        mu_k_loop = local_pert_mu_biased.copy()
        if affected_mask_loop.sum() > 0:
            effect_directions = rng.choice(
                [effect_factor, 1.0 / effect_factor], size=affected_mask_loop.sum()
            )  # alpha in Eq 2 and 4
            mu_k_loop[affected_mask_loop] *= effect_directions

        lib_size_pert = rng.lognormal(
            mean=mu_l, sigma=0.1714, size=Nk
        )  # 0.1714 from all cells of the Norman19 dataset
        pert_counts = sample_nb_counts(
            mean=mu_k_loop, l_c=lib_size_pert, theta=local_all_theta, rng=rng
        )
        counts_blocks.append(pert_counts)
        obs_blocks.append(
            build_obs_block(
                n_cells=Nk,
                perturbation_id=perturbation_id,
                perturbation_name=str(perturbation_names[perturbation_id]),
                cell_line=0,
            )
        )

    return build_synthetic_adata(
        counts_blocks=counts_blocks,
        obs_blocks=obs_blocks,
        var=var,
        normalize=normalize,
        normalized_layer_key=normalized_layer_key,
    ), all_affected_masks
