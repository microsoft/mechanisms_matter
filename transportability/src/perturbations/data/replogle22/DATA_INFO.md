# Replogle22 Processed Data

The preprocessing in [get_data.py](get_data.py) creates three pair-specific datasets with K562 as the shared anchor cell line.

## Pipeline

For every selected partner, the script:

1. Loads K562 and the partner raw single-cell H5AD files.
2. Harmonizes Jurkat/HepG2 metadata and converts their Ensembl-indexed features to unique gene symbols.
3. Intersects measured genes across the pair with `AnnCollection(join_vars="inner")`.
4. Retains perturbations represented by at least `64` cells in both cell lines.
5. Filters cells with fewer than `200` genes and genes observed in fewer than `3` cells.
6. Stores raw counts in `adata.layers["counts"]`.
7. Normalizes each cell to `1e4`, applies `log1p`, and retains up to `8192` highly variable genes.
8. Removes perturbations whose target genes are absent from the final feature set.
9. Computes raw-count mean/variance summaries, PCA, neighbors, UMAP, and DEG summaries.
10. Writes the processed data, gene information, DEG tables, and a manifest.

All thresholds are configurable through the preprocessing CLI. RPE1 uses the same processing behavior as the previous two-cell-line script.

## Output Layout

```text
data/replogle22/
  RPE1/
    processed.h5ad
    genes.csv.gz
    names_df_vsrest.pkl
    scores_df_vsrest.pkl
    names_df_vsctrl.pkl
    scores_df_vsctrl.pkl
    processed_manifest.json
  Jurkat/
    ...
  HepG2/
    ...
```

The manifest in each directory records source files, source releases, raw shapes, shared-gene counts, filtering parameters, final dimensions, and artifact paths.

## AnnData Layout

- `adata.X`: library-size-normalized and `log1p`-transformed expression.
- `adata.layers["counts"]`: raw counts after cell/gene filtering.
- `adata.obs["perturbation"]`: perturbation target, with non-targeting cells represented as `control`.
- `adata.obs["condition"]`: copy of `perturbation`.
- `adata.obs["cell_line"]`: `K562` and the selected partner.
- `adata.var_names`: shared gene symbols retained after filtering and HVG selection.
- `adata.uns`: DEG dictionaries and per-condition count summaries.
