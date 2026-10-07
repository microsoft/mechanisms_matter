"""Shared preprocessing helpers for perturbation datasets."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import scanpy as sc
from tqdm import tqdm


def write_manifest_json(manifest_path, manifest):
    """Write the manifest dictionary to JSON and return it."""
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def perturbation_targets_in_gene_set(
    label: str,
    gene_name_set: set[str],
    control_label: str = "control",
) -> bool:
    """
    Check if all perturbation targets in the label are present in the gene name set.

    Args:
        label: Perturbation label, e.g. "geneA+geneB" or "control"
        gene_name_set: Set of valid gene names
        control_label: Label used for control group (default: "control")

    Returns:
        bool: True if all targets are in the gene name set (or if label is the control), False otherwise
    """
    if str(label) == control_label:
        return True
    targets = [token for token in str(label).split("+") if token]
    return all(target in gene_name_set for target in targets)


def compute_degs(adata, mode="vsrest", pval_threshold=0.05):
    """
    Compute differentially expressed genes (DEGs) for each perturbation.

    Args:
        adata: AnnData object with processed data
        mode: 'vsrest' or 'vscontrol'
            - 'vsrest': Compare each perturbation vs all other perturbations (excluding control)
            - 'vscontrol': Compare each perturbation vs control only
        pval_threshold: P-value threshold for significance (default: 0.05)

    Returns:
        dict: rank_genes_groups results dictionary

    Adds to adata.uns:
        - deg_dict_{mode}: Dictionary with perturbation as key and dict with 'up'/'down' DEGs as values
        - rank_genes_groups_{mode}: Full rank_genes_groups results
    """
    if mode == "vsrest":
        # Remove control cells for vsrest analysis
        adata_subset = adata[adata.obs["condition"] != "control"].copy()
        reference = "rest"
    elif mode == "vscontrol":
        # Use full dataset for vscontrol analysis
        adata_subset = adata.copy()
        reference = "control"
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")

    # Compute DEGs
    sc.tl.rank_genes_groups(
        adata_subset, "condition", method="t-test_overestim_var", reference=reference
    )

    # Extract results
    names_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["pvals_adj"])
    logfc_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["logfoldchanges"])

    # For each perturbation, get the significant DEGs up and down regulated
    deg_dict = {}
    for pert in tqdm(adata_subset.obs["condition"].unique(), desc=f"Computing DEGs {mode}"):
        if mode == "vscontrol" and pert == "control":
            continue  # Skip control when comparing vs control

        pert_degs = names_df[pert]
        pert_pvals = pvals_adj_df[pert]
        pert_logfc = logfc_df[pert]

        # Get significant DEGs
        significant_mask = pert_pvals < pval_threshold
        pert_degs_sig = pert_degs[significant_mask]
        pert_logfc_sig = pert_logfc[significant_mask]

        # Split into up and down regulated
        pert_degs_sig_up = pert_degs_sig[pert_logfc_sig > 0].tolist()
        pert_degs_sig_down = pert_degs_sig[pert_logfc_sig < 0].tolist()

        deg_dict[pert] = {"up": pert_degs_sig_up, "down": pert_degs_sig_down}

    # Save results to adata.uns
    adata.uns[f"deg_dict_{mode}"] = deg_dict
    adata.uns[f"rank_genes_groups_{mode}"] = adata_subset.uns["rank_genes_groups"].copy()

    return adata_subset.uns["rank_genes_groups"]


def write_processed_manifest(
    manifest_path,
    adata,
    raw_data_paths,
    processed_h5ad_path,
    genes_csv_path,
    artifact_paths=None,
    extra_metadata=None,
):
    """
    Write a compact JSON summary for a processed AnnData dataset.

    Args:
        manifest_path: Output JSON path.
        adata: Processed AnnData object or backed AnnData view.
        raw_data_paths: Iterable of raw/downloaded source file paths.
        processed_h5ad_path: Saved processed h5ad path.
        genes_csv_path: Saved gene metadata CSV path.
        artifact_paths: Optional mapping of additional artifact labels to paths.
        extra_metadata: Optional mapping of dataset-specific summary fields.

    Returns:
        dict: The manifest payload written to disk.
    """
    manifest_path = Path(manifest_path).resolve()
    processed_h5ad_path = Path(processed_h5ad_path).resolve()
    genes_csv_path = Path(genes_csv_path).resolve()

    raw_paths = [str(Path(path).resolve()) for path in raw_data_paths]
    artifact_paths = artifact_paths or {}
    extra_metadata = extra_metadata or {}
    perturbation_column = next(
        (column for column in ("condition", "perturbation") if column in adata.obs.columns),
        None,
    )
    control_mask = None
    if perturbation_column is not None:
        perturbation_labels = adata.obs[perturbation_column]
        control_mask = (
            perturbation_labels.astype(str).str.strip().str.lower().isin({"control", "ctrl"})
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_paths": raw_paths,
        "processed_dir": str(processed_h5ad_path.parent),
        "processed_h5ad": str(processed_h5ad_path),
        "n_raw_datasets": len(raw_paths),
        "n_obs_total": int(adata.n_obs),
        "n_obs_control": int(control_mask.sum()) if control_mask is not None else None,
        "n_obs_pert": int((~control_mask).sum()) if control_mask is not None else None,
        "n_vars_final": int(adata.n_vars),
        "n_perturbations_final": (
            int(adata.obs["perturbation"].astype(str).nunique())
            if "perturbation" in adata.obs.columns
            else None
        ),
        "n_obs_by_cell_line": (
            {
                str(cell_line): int(count)
                for cell_line, count in adata.obs["cell_line"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            }
            if "cell_line" in adata.obs.columns
            else None
        ),
        "obs_columns": [str(column) for column in adata.obs.columns.tolist()],
        "var_columns": [str(column) for column in adata.var.columns.tolist()],
        "layers": sorted(str(key) for key in adata.layers.keys()),
        "obsm_keys": sorted(str(key) for key in adata.obsm.keys()),
        "obsp_keys": sorted(str(key) for key in adata.obsp.keys()),
        "varm_keys": sorted(str(key) for key in adata.varm.keys()),
        "uns_keys": sorted(str(key) for key in adata.uns.keys()),
        "genes_csv": str(genes_csv_path),
    }
    manifest.update(
        {key: str(Path(path).resolve()) for key, path in sorted(artifact_paths.items())}
    )
    manifest.update(extra_metadata)

    return write_manifest_json(manifest_path, manifest)
