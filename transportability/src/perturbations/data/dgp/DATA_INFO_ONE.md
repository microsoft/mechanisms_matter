# Direct DGP Output Schema

`directDGP(...)` now returns a single in-memory `AnnData` object plus the per-perturbation affected-gene masks:

```python
adata, affected_masks = directDGP(...)
```

## Return values

- `adata: anndata.AnnData`
- `affected_masks: list[np.ndarray]`

Each entry in `affected_masks` is a boolean mask of length `G` for one perturbation.

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
  Integer context label. For `directDGP`, this is always `0`.
- `adata.var_names`
  The sampled gene names used in the simulation.

## Notes

- Raw counts live in `.X`; there is no duplicate `layers["counts"]`.
