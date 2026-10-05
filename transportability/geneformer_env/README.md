# Geneformer Environment

Isolated environment for extracting Geneformer cell embeddings. It uses
Transformers 4.57.6; the main transportability environment uses 5.16.1. Geneformer
code is pinned to commit `04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5`.

## Setup

```bash
cd transportability/geneformer_env
GIT_LFS_SKIP_SMUDGE=1 uv --no-config sync --locked --python 3.11
```

Use uv 0.11.8 and Python 3.11. Each environment has its own `uv.lock` and virtual
environment. On Alliance HPC, perform setup and extraction inside Slurm with
`UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, `HF_HOME`, and `XDG_CACHE_HOME` under
`$SCRATCH`; see the [environment guide](../../docs/environments.md).

## Usage

### 1. Fetch model weights + dictionaries

```bash
uv run --locked python fetch_geneformer_assets.py --model Geneformer-V2-104M --dest ./gf_assets
```

This writes the model to `./gf_assets/Geneformer-V2-104M/` and Geneformer's `*.pkl`
dictionaries to `./gf_assets/geneformer/`. Both are required: `uv sync` runs with
`GIT_LFS_SKIP_SMUDGE=1`, so the dictionaries bundled inside the installed package are
git-LFS pointer stubs, not real pickles.

### 2. Extract embeddings

```bash
uv run --locked python extract_geneformer_embeddings.py \
  --input ../src/perturbations/data/norman19/norman19_processed.h5ad \
  --output ../src/perturbations/data/norman19/norman19_geneformer.h5ad \
  --counts-layer counts \
  --model-dir ./gf_assets/Geneformer-V2-104M \
  --obsm-key X_geneformer
```

This writes cell embeddings to `adata.obsm["X_geneformer"]` in the output file.

The tokenizer dictionaries and the gene-symbol → Ensembl map are auto-discovered from
the directory holding `--model-dir` (i.e. the `./gf_assets` layout above). Pass
`--dictionary-dir` if your assets live elsewhere.

**Devices**: the lock installs PyTorch 2.14.0 with CUDA 13.0 on Linux and CPU
PyTorch on other platforms. Linux GPU extraction needs an NVIDIA GPU and a driver
compatible with that CUDA runtime. Geneformer hardcodes CUDA for its input tensors;
the extraction script installs a CPU shim when no GPU is present. CPU extraction
is slow; use a GPU for real datasets.

### 3. Use embeddings in the main env

Back in the main transportability environment, run real experiments with the enriched `.h5ad`:

```bash
cd ../src/perturbations
uv run --locked python -m perturbations.analyses.real_experiments.run \
  --dataset_name norman19 \
  --dataset_path data/norman19/norman19_geneformer.h5ad \
  --split_strategy in-context \
  --n_trials 5 \
  --basal_embedding_key X_geneformer \
  --model STATE
```

`--basal_embedding_key` tells the STATE model to read `test_adata.obsm["X_geneformer"]` as its encoder input instead of raw gene expression.

## How it connects

The main package (`state_gene`) reads the precomputed embeddings via `basal_embedding_key`. It never imports anything from `geneformer_env/` directly.
