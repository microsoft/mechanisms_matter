"""Download and preprocess Norman19 perturbation data into project artifacts."""

import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm

from ..preprocess import (
    compute_degs,
    perturbation_targets_in_gene_set,
    write_processed_manifest,
)

rng = np.random.default_rng(42)

data_url = "https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1"
data_cache_dir = Path("./data/norman19")

if not data_cache_dir.exists():
    data_cache_dir.mkdir(parents=True)

tmp_data_dir = data_cache_dir / "norman19_downloaded.h5ad"

if not tmp_data_dir.exists():
    sp.call(f"wget -q {data_url} -O {tmp_data_dir}", shell=True)

adata = sc.read_h5ad(tmp_data_dir)

MIN_CELLS_PER_PERT_PER_CELLLINE = 256
MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3
NORMALIZE_TARGET_SUM = 1e4
HVG_TOP_GENES = 8192
ct = adata.obs.pivot_table(index="perturbation", aggfunc="size", fill_value=0).sort_index()
valid = ct.iloc[:] >= MIN_CELLS_PER_PERT_PER_CELLLINE
print(f"Number of perturbations with >= {MIN_CELLS_PER_PERT_PER_CELLLINE} cells: {valid.sum()}")
print(f"Number of perturbations in original data: {adata.obs['perturbation'].nunique()}")


# Rename columns
adata.obs = adata.obs.rename(
    columns={
        "nCount_RNA": "ncounts",
        "nFeature_RNA": "ngenes",
        "percent.mt": "percent_mito",
    }
)
adata.obs["perturbation"] = adata.obs["perturbation"].str.replace("_", "+")
adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
adata.obs["condition"] = adata.obs.perturbation.copy()
adata.X = csr_matrix(adata.X)

# Filter cells
sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)

# Stash raw counts
adata.layers["counts"] = adata.X.copy()

# Do library size norm and log1p
sc.pp.normalize_total(adata, target_sum=NORMALIZE_TARGET_SUM)
sc.pp.log1p(adata)

# Get 8192 HVGs -- subset the adata object to only include the HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=HVG_TOP_GENES, subset=True)

# Keep only perturbations whose target genes all remain in the final feature set.
final_gene_set = set(adata.var_names.astype(str))
perturbation_labels = adata.obs["perturbation"].astype(str)
keep_mask = perturbation_labels.map(
    lambda label: perturbation_targets_in_gene_set(label, final_gene_set)
).to_numpy()
if not keep_mask.all():
    adata = adata[keep_mask].copy()
    adata.obs["perturbation"] = adata.obs["perturbation"].cat.remove_unused_categories()
    adata.obs["condition"] = adata.obs["perturbation"].copy()

# For every kept perturbation, for every kept gene, calculate the mean and variance of the raw counts.
mean_df = pd.DataFrame(index=adata.var_names, columns=adata.obs["condition"].unique())
disp_df = pd.DataFrame(index=adata.var_names, columns=adata.obs["condition"].unique())
condition_labels = adata.obs["condition"].astype(str).to_numpy()
for pert in tqdm(adata.obs["condition"].unique()):
    pert_mask = condition_labels == str(pert)
    pert_counts = adata.layers["counts"][pert_mask].toarray()
    mean_df.loc[:, pert] = np.mean(pert_counts, axis=0)
    disp_df.loc[:, pert] = np.var(pert_counts, axis=0)

# Save to the uns dictionary
mean_df_dict = mean_df.to_dict(orient="list")
disp_df_dict = disp_df.to_dict(orient="list")
adata.uns["mean_dict"] = mean_df_dict
adata.uns["disp_dict"] = disp_df_dict
adata.uns["mean_disp_dict_genes"] = disp_df.index.tolist()

# Do PCA
sc.pp.pca(adata)

# Do UMAP
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Calculate DEGs between each perturbation and all other perturbations
_ = compute_degs(adata, mode="vsrest")

# Calculate DEGs with respect to the control perturbation
_ = compute_degs(adata, mode="vscontrol")

# Convert to format that can be saved
SCORE_TYPE = "scores"  # or 'logfoldchanges'
names_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"]["names"])
scores_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"][SCORE_TYPE])
names_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"]["names"])
scores_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"][SCORE_TYPE])
# logfc_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"]["logfoldchanges"])

# Save dataframes to csv
names_df_vsrest.to_pickle(f"{data_cache_dir}/norman19_names_df_vsrest.pkl")
scores_df_vsrest.to_pickle(f"{data_cache_dir}/norman19_scores_df_vsrest.pkl")
names_df_vsctrl.to_pickle(f"{data_cache_dir}/norman19_names_df_vsctrl.pkl")
scores_df_vsctrl.to_pickle(f"{data_cache_dir}/norman19_scores_df_vsctrl.pkl")
# logfc_df_vsctrl.to_pickle(f'{data_cache_dir}/norman19_logfc_df_vsctrl.pkl')

# Remove these from the adata object
adata.uns.pop("rank_genes_groups_vsrest", None)
adata.uns.pop("rank_genes_groups_vscontrol", None)
adata.uns.pop("rank_genes_groups", None)


# Save the data
output_data_path = f"{data_cache_dir}/norman19_processed.h5ad"
adata.write_h5ad(output_data_path)
# Save adata.var to a CSV
genes_path = f"{data_cache_dir}/norman19_genes.csv.gz"
adata.var.to_csv(genes_path)

manifest_path = f"{data_cache_dir}/processed_manifest.json"
write_processed_manifest(
    manifest_path,
    adata=adata,
    raw_data_paths=[tmp_data_dir],
    processed_h5ad_path=output_data_path,
    genes_csv_path=genes_path,
    artifact_paths={
        "names_df_vsrest_pickle": f"{data_cache_dir}/norman19_names_df_vsrest.pkl",
        "scores_df_vsrest_pickle": f"{data_cache_dir}/norman19_scores_df_vsrest.pkl",
        "names_df_vscontrol_pickle": f"{data_cache_dir}/norman19_names_df_vsctrl.pkl",
        "scores_df_vscontrol_pickle": f"{data_cache_dir}/norman19_scores_df_vsctrl.pkl",
    },
    extra_metadata={
        "n_perturbations_original": int(ct.shape[0]),
        "n_perturbations_after_min_cell_filter": int(valid.sum()),
        "min_cells_per_perturbation": int(MIN_CELLS_PER_PERT_PER_CELLLINE),
        "cell_filter_min_genes": int(MIN_GENES_PER_CELL),
        "gene_filter_min_cells": int(MIN_CELLS_PER_GENE),
        "normalize_target_sum": float(NORMALIZE_TARGET_SUM),
        "hvg_top_genes": int(HVG_TOP_GENES),
    },
)
print(f"Wrote processed manifest to {manifest_path}")
