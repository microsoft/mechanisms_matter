# CD4+ Processed Data (`AnnCollection` Backed)

This describes the processed CD4+ dataset produced by [`process_data.py`](process_data.py), where data are saved as multiple chunked `.h5ad` files and loaded with `AnnCollection` in backed mode.

Primary processed artifacts:

- `data/cd4+/processed/processed_manifest.json`
- `data/cd4+/processed/chunks/cd4_processed_*.h5ad` (12)
- `data/cd4+/processed/cd4_genes.csv.gz` (full gene metadata table)

From `processed_manifest.json`:

- `n_chunks = 12`
- `n_obs_total = 2,299,062` cells
- `n_vars_final = 3,463` genes

## How to load (backed `AnnCollection`)

The backed loading logic is:

```python
import anndata as ad
from anndata.experimental import AnnCollection

backed_chunks = [ad.read_h5ad(path, backed="r") for path in chunk_paths]
collection = AnnCollection(
    backed_chunks,
    join_vars="inner",
    label="chunk_id",
    keys=[path.stem for path in chunk_paths],
    index_unique="-",
)
```

Key collection-level properties:

- `collection.shape = (2299062, 3463)` (from manifest; full load is large)
- `collection.obs` includes chunk-level metadata + appended `chunk_id`
- `collection.var_names` available (gene IDs)
- `collection` has no `.var` attribute in current `anndata.experimental` API
- `collection.attrs_keys` reports:
  - `obs`: 18 fields (listed below)
  - `layers`: `["counts"]`
  - `obsm`: `[]`

## Per-chunk `AnnData` structure (`ad.read_h5ad(..., backed="r")`)

Each chunk has consistent schema (validated across representative chunks):

- Shape per file: `(<=250000, 3463)`
- `adata.X`
  - Type (backed): `_CSRDataset`
  - Dtype: `float32`
  - Contents: normalized (`target_sum=1e4`) + `log1p` expression
- `adata.layers["counts"]`
  - Type: `scipy.sparse.csr_matrix`
  - Dtype: `float32`
  - Contents: pre-normalization counts

### `adata.obs` (17 columns)

- `lane_id` (`category`)
- `n_genes_by_counts` (`int32`)
- `total_counts` (`float32`)
- `pct_counts_mt` (`float32`)
- `top_guide_UMI_counts` (`float64`)
- `guide_id` (`category`)
- `perturbed_gene_name` (`category`)
- `perturbed_gene_id` (`category`)
- `guide_type` (`category`)
- `PuroR` (`float32`)
- `guide_group` (`category`)
- `low_quality` (`bool`)
- `perturbation` (`category`) (`perturbed_gene_id` for targeting cells, or `control`)
- `condition` (`category`) (copy of `perturbation`)
- `donor` (`category`)
- `timepoint` (`category`)
- `context` (`category`)

### `adata.var` (12 columns)

- `gene_ids` (`object`)
- `feature_types` (`category`)
- `genome` (`category`)
- `gene_name` (`object`)
- `mt` (`bool`)
- `n_cells_after_cell_qc` (`int64`)
- `total_counts_after_cell_qc` (`float64`)
- `passes_gene_qc` (`bool`)
- `hvg_rest` (`bool`)
- `hvg_stim8hr` (`bool`)
- `hvg_stim48hr` (`bool`)
- `highly_variable` (`bool`)

Other slots:

- `adata.uns`: `["log1p", "source_context", "source_file"]`
- `adata.obsm`: empty
- `adata.obsp`: empty
- `adata.varm`: empty
- `adata.raw is None`
- `adata.var_names` index name: `gene_id`

## When materializing from `AnnCollection`

If you slice and materialize, e.g. `collection[idx].to_adata()`:

- `X` becomes in-memory `csr_matrix` (`float32`)
- `layers["counts"]` is preserved
- `obs` contains the 17 chunk columns plus `chunk_id` (18 total)
- `var` keeps gene index (`gene_id`) but has no var columns (`shape = (3463, 0)`)
- `uns` is empty

So for gene-level metadata after materialization, use:

- `data/cd4+/processed/cd4_genes.csv.gz`
  - or read `var` from one chunk file directly.
