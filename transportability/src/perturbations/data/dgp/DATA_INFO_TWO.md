# Causal DGP Output Schema

`causalDGP(...)` now returns a single in-memory `AnnData` object plus the affected-gene masks for both cell-line-specific systems:

```python
adata, affected_masks_pair = causalDGP(...)
```

## Return values

- `adata: anndata.AnnData`
- `affected_masks_pair: list[list[np.ndarray]]`

`affected_masks_pair[0]` corresponds to the base causal system and `affected_masks_pair[1]` corresponds to the altered causal system.

## AnnData schema

- `adata.X`
  Raw counts as a CSR sparse matrix with shape `(N0 + P * Nk, G)`.
- `adata.layers["normalized_log1p"]`
  Normalized and `log1p`-transformed expression matrix with the same shape as `adata.X`.
- `adata.obs["perturbation"]`
  String labels. Control cells use `"control"`.
- `adata.obs["perturbation_id"]`
  Integer labels. Control cells use `-1`, perturbations use `0..P-1`.
- `adata.obs["cell_line"]`
  Binary context label taking values `0` or `1`.
- `adata.var_names`
  The sampled gene names used in the simulation.

## Notes

- Raw counts live in `.X`; there is no duplicate `layers["counts"]`.
- `output_dir` is only used when `visualize=True`, in which case diagnostic figures are written there.
