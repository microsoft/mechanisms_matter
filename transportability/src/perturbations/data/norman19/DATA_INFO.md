# Norman19 Processed Data

This dataset is produced by [get_data.py](get_data.py) and summarized in [processed_manifest.json](processed_manifest.json).

## Pipeline Summary

The preprocessing script:

1. Downloads the Norman19 source `.h5ad` file from Zenodo.
2. Renames selected observation columns:
   - `nCount_RNA -> ncounts`
   - `nFeature_RNA -> ngenes`
   - `percent.mt -> percent_mito`
3. Rewrites perturbation labels by replacing `_` with `+`.
4. Filters cells with fewer than `200` genes and genes observed in fewer than `3` cells.
5. Stores raw counts in `adata.layers["counts"]`.
6. Applies library-size normalization with `target_sum=1e4`, then `log1p`.
7. Restricts the feature space to the top `8192` highly variable genes.
8. Drops perturbations whose target genes are no longer fully present after HVG subsetting.
9. Computes per-perturbation raw-count mean and variance summaries and stores them in `adata.uns`.
10. Computes PCA, neighbors, UMAP, and DEG summaries (`vsrest` and `vscontrol`).
11. Writes the processed `.h5ad`, gene metadata CSV, DEG pickles, and a manifest JSON.

## Output Artifacts

Main outputs written under `data/norman19/`:

- `norman19_processed.h5ad`
- `norman19_genes.csv.gz`
- `norman19_names_df_vsrest.pkl`
- `norman19_scores_df_vsrest.pkl`
- `norman19_names_df_vsctrl.pkl`
- `norman19_scores_df_vsctrl.pkl`
- `processed_manifest.json`

## Manifest Summary

From [processed_manifest.json](processed_manifest.json):

- Raw input datasets: `1`
- Final shape: `adata.shape = (83597, 8192)`
- Control cells: `11855`
- Perturbed cells: `71742`
- Final perturbation labels: `157` total
- Final cell lines: `{"K562": 83597}`
- Original perturbations before filtering: `237`
- Perturbations passing the minimum-cell filter: `176`
- Minimum cells per perturbation: `256`
- Cell filter: `min_genes=200`
- Gene filter: `min_cells=3`
- Normalization target sum: `10000`
- HVGs retained: `8192`

The drop from `176` perturbations after the minimum-cell filter to `157` final perturbations happens because the script removes perturbations whose target genes are not fully retained in the post-HVG feature set.

## `AnnData` Layout

- `adata.X`
  Normalized, `log1p`-transformed expression matrix.
- `adata.layers["counts"]`
  Raw count matrix after cell/gene filtering and before normalization.
- `adata.obs_names`
  Unique cell barcodes.
- `adata.var_names`
  Unique gene symbols for the retained HVG feature set.

## `adata.obs`

The manifest records these observation columns:

- `guide_id`
- `read_count`
- `UMI_count`
- `coverage`
- `gemgroup`
- `good_coverage`
- `number_of_cells`
- `tissue_type`
- `cell_line`
- `cancer`
- `disease`
- `perturbation_type`
- `celltype`
- `organism`
- `perturbation`
- `nperts`
- `ngenes`
- `ncounts`
- `percent_mito`
- `percent_ribo`
- `condition`
- `n_genes`

Notes:

- `perturbation` is the main perturbation label used throughout the project.
- `condition` is a copy of `perturbation`.
- `cell_line` has a single retained value: `K562`.

## `adata.var`

The manifest records these variable columns:

- `ensemble_id`
- `ncounts`
- `ncells`
- `n_cells`
- `highly_variable`
- `means`
- `dispersions`
- `dispersions_norm`

## Embeddings and Graph Slots

The processed object contains:

- `adata.obsm["X_pca"]`
- `adata.obsm["X_umap"]`
- `adata.obsp["connectivities"]`
- `adata.obsp["distances"]`
- `adata.varm["PCs"]`

## `adata.uns`

The manifest records these unstructured keys:

- `deg_dict_vscontrol`
- `deg_dict_vsrest`
- `disp_dict`
- `hvg`
- `log1p`
- `mean_dict`
- `mean_disp_dict_genes`
- `neighbors`
- `pca`
- `umap`

Additional DEG ranking tables are written as separate pickle artifacts rather than being kept in `adata.uns`.
