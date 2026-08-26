"""Generate realistic synthetic perturbation data with context-dependent effects."""

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def generate_synthetic_perturbation_data(
    n_cells: int = 3000,
    n_genes: int = 200,
    n_perturbations: int = 5,
    context_key: str = "cell_type",
    control_label: str = "control",
    perturbation_column: str = "perturbation",
    seed: int = 42,
) -> anndata.AnnData:
    """
    Generate synthetic AnnData with context-dependent perturbation effects.

    Args:
        n_cells: Total number of cells to generate.
        n_genes: Number of genes.
        n_perturbations: Number of distinct perturbations (excluding control).
        context_key: Column name for context labels in .obs (e.g. "cell_type").
        control_label: Label for unperturbed control cells.
        perturbation_column: Column name for perturbation labels in .obs.
        seed: Random seed for reproducibility.

    Returns:
        AnnData with .X as sparse counts, raw counts in .layers["counts"],
        perturbation labels in .obs[perturbation_column], and context in .obs["cell_type"].
    """
    rng = np.random.default_rng(seed)

    contexts = ["type_A", "type_B"]

    gene_names = [f"gene_{i}" for i in range(n_genes)]
    pert_names = [f"gene_{i}" for i in range(n_perturbations)]

    # Assign cells to contexts roughly equally
    cell_contexts: np.ndarray = rng.choice(contexts, size=n_cells)

    # Assign perturbations (including control)
    perturbations: np.ndarray = rng.choice(
        [*pert_names, control_label],
        size=n_cells,
    )

    # --- Context-specific baseline expression ---
    context_baselines: dict[str, np.ndarray] = {}
    for ctx in contexts:
        baseline = rng.exponential(2.0, size=n_genes)
        ctx_idx = contexts.index(ctx)
        module_start = ctx_idx * 30
        module_end = module_start + 30
        baseline[module_start:module_end] *= 3.0
        context_baselines[ctx] = baseline

    # --- Context-specific perturbation effects ---
    pert_effects: dict[tuple[str, str], np.ndarray] = {}
    for pert in pert_names:
        pert_idx = pert_names.index(pert)
        shared_effect = np.zeros(n_genes)
        shared_effect[pert_idx] = rng.uniform(30, 60)

        for ctx in contexts:
            effect = np.array(shared_effect, dtype=np.float64)
            ctx_idx = contexts.index(ctx)
            n_secondary = int(rng.integers(3, 6))
            secondary_genes = rng.choice(
                np.arange(n_perturbations, n_genes),
                size=n_secondary,
                replace=False,
            )
            for g in secondary_genes:
                effect[g] = rng.uniform(10, 40) * (1 if ctx_idx == 0 else -0.5)

            cascade_start = 100 + ctx_idx * 20 + pert_idx * 4
            cascade_end = min(cascade_start + 4, n_genes)
            effect[cascade_start:cascade_end] = rng.uniform(
                15, 35, size=cascade_end - cascade_start
            )

            pert_effects[(pert, ctx)] = effect

    # --- Generate counts ---
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        ctx = str(cell_contexts[i])
        baseline = context_baselines[ctx]

        for g in range(n_genes):
            mu = baseline[g]
            n_param = 5.0
            p_param = n_param / (n_param + mu)
            counts[i, g] = rng.negative_binomial(n_param, p_param)

        pert = str(perturbations[i])
        if pert != control_label:
            effect = pert_effects[(pert, ctx)]
            for g in range(n_genes):
                if effect[g] > 0:
                    counts[i, g] += rng.poisson(effect[g])
                elif effect[g] < 0:
                    reduction = rng.poisson(int(-effect[g]))
                    counts[i, g] = max(0.0, counts[i, g] - reduction)

    adata = anndata.AnnData(
        X=csr_matrix(counts),
        obs={
            perturbation_column: perturbations,
            context_key: pd.Categorical(cell_contexts),
        },
    )
    adata.var_names = gene_names
    adata.layers["counts"] = adata.X.copy()  # type: ignore[reportUnknownMemberType]
    return adata
