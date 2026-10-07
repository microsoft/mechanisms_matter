# Reproducible environments

Use Python 3.11 and uv 0.11.8 as the reference setup. Each environment has its
own `pyproject.toml` and `uv.lock`. The lockfiles record exact package versions,
artifact hashes, platform markers, and the Geneformer Git revision. Runtime
packages come from public PyPI; Geneformer PyTorch comes from the explicit public
PyTorch indexes.

| Environment | Manifest and lock location | Purpose |
| --- | --- | --- |
| Main package | `transportability/` | Simulators, metrics, preprocessing, model dependencies, and tests |
| Geneformer | `transportability/geneformer_env/` | Isolated Geneformer embedding extraction |

Geneformer uses Transformers 4.x, while the main package uses Transformers 5.x,
so Geneformer has a separately locked environment.

## Installation and validation

For the main package, from the repository root:

```bash
cd transportability
uv --no-config sync --locked --python 3.11
uv --no-config run --locked python -c "import perturbations, scanpy, scvi, torch"
uv --no-config run --locked pytest -q
```

For Geneformer, run this from the repository root:

```bash
cd transportability/geneformer_env
GIT_LFS_SKIP_SMUDGE=1 uv --no-config sync --locked --python 3.11
uv --no-config run --locked python -c "import geneformer, transformers, torch; print(torch.__version__, torch.version.cuda)"
```

`--no-config` avoids inheriting machine-specific uv configuration. `--locked`
checks manifest/lock agreement and prevents implicit dependency updates; see
[uv's locking documentation](https://docs.astral.sh/uv/concepts/projects/sync/).
Use `uv run --locked` for experiment commands too. A plain `pip install -e .`
does not use `uv.lock` and is not the reference environment.

Each project creates its own `.venv` by default. Do not use the Geneformer
environment for main-package experiments. Save Geneformer embeddings in an
`.h5ad` file under `obsm["X_geneformer"]`, then load that file in the main
package, as described in the
[Geneformer guide](../transportability/geneformer_env/README.md).

## External artifacts

Data, model weights, and Geneformer tokenizer dictionaries are external artifacts,
separate from these package locks. `GIT_LFS_SKIP_SMUDGE=1` prevents Git LFS model
assets from being downloaded while installing Geneformer. Fetch assets using the
Geneformer guide before extraction.
