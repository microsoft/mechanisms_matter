# Geneformer Environment

Isolated environment for extracting Geneformer cell embeddings. Kept separate because Geneformer pins `transformers<5`, which conflicts with the main transportability env.

## Setup

```bash
cd transportability/geneformer_env
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## Usage

### 1. Fetch model weights + dictionaries

```bash
uv run python fetch_geneformer_assets.py --model Geneformer-V2-104M --dest ./gf_assets
```

This writes the model to `./gf_assets/Geneformer-V2-104M/` and Geneformer's `*.pkl`
dictionaries to `./gf_assets/geneformer/`. Both are required: `uv sync` runs with
`GIT_LFS_SKIP_SMUDGE=1`, so the dictionaries bundled inside the installed package are
git-LFS pointer stubs, not real pickles.

### 2. Extract embeddings

```bash
uv run python extract_geneformer_embeddings.py \
  --input ../data/norman19/norman19_processed.h5ad \
  --output ../data/norman19/norman19_geneformer.h5ad \
  --counts-layer counts \
  --model-dir ./gf_assets/Geneformer-V2-104M \
  --obsm-key X_geneformer
```

This writes cell embeddings to `adata.obsm["X_geneformer"]` in the output file.

The tokenizer dictionaries and the gene-symbol → Ensembl map are auto-discovered from
the directory holding `--model-dir` (i.e. the `./gf_assets` layout above). Pass
`--dictionary-dir` if your assets live elsewhere.

**Devices**: `pyproject.toml` installs a CPU build of torch by default, and Geneformer
hardcodes CUDA for its input tensors. The script installs a CPU shim automatically when
no GPU is present, so extraction works either way — but CPU extraction is slow, so use a
CUDA torch build (see the commented index in `pyproject.toml`) for real datasets.

### 3. Use embeddings in the main env

Back in the main transportability environment, run real experiments with the enriched `.h5ad`:

```bash
cd transportability
uv run python -m perturbations.analyses.real_experiments.run \
  --dataset_name norman19 \
  --dataset_path data/norman19/norman19_geneformer.h5ad \
  --split_strategy in-context \
  --n_trials 5 \
  --basal_embedding_key X_geneformer
```

`--basal_embedding_key` tells the STATE model to read `test_adata.obsm["X_geneformer"]` as its encoder input instead of raw gene expression.

## How it connects

The main package (`state_gene`) reads the precomputed embeddings via `basal_embedding_key`. It never imports anything from `geneformer_env/` directly.
