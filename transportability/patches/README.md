The local STATE backend is intentionally excluded from Git. `state_gene_cd4.patch`
records the CD4 Geneformer runtime changes against the backend available in this
workspace on 2026-09-13. It preserves the existing training hyperparameters and
evaluation behavior while sharing context class IDs across splits, seeding each
trial, and saving resumable checkpoints.

From the repository root, inspect and apply it to the corresponding local backend:

```bash
git apply --check transportability/patches/state_gene_cd4.patch
git apply transportability/patches/state_gene_cd4.patch
```

For an already patched backend, `git apply --reverse --check` verifies that the
patch is present. The focused regression file is
`transportability/test/models/test_state_gene_runtime.py`; run it through Slurm
with the project's Python environment.

The runner gives every seed a stable checkpoint directory. `last.ckpt` contains
the model, optimizer, Lightning loops and callbacks, sampling RNGs, and hashes of
the exact split data and embeddings. Resuming rejects mismatched data, model
settings, and seeds. A checkpoint saved after training completes can be reused
to regenerate predictions after a downstream evaluation interruption.
