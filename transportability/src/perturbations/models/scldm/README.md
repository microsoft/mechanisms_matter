# scLDM — Single-Cell Latent Diffusion Model

A latent flow-matching / diffusion model for generating single-cell RNA-seq
perturbation profiles. scLDM couples a transformer **VAE** (tokenizer) with a
**DiT** (Diffusion Transformer) trained via flow matching in the VAE latent
space, with classifier-free guidance over perturbation conditions.

This is a Hydra-free port targeting the Perturb-seq dataset, adapted from
[czi-ai/scldm](https://github.com/czi-ai/scldm).

## Architecture

scLDM is trained in two stages and sampled in a third:

```
                 ┌──────────────────────────────────────┐
   counts ──────▶│ TransformerVAE (encoder → z → decoder)│──▶ NB reconstruction
                 └──────────────────────────────────────┘
                            │ z (frozen after stage 2)
                            ▼
                 ┌──────────────────────────────────────┐
   condition ───▶│ DiT + flow-matching transport (LDM)  │──▶ sampled latents z*
   (cell_line,   └──────────────────────────────────────┘
    gene)                   │ decode(z*, size factors)
                            ▼
                     generated counts
```

- **Stage 2 — VAE**: a transformer encoder/decoder with inducing-point
  cross-attention and a Negative Binomial decoder head compresses counts into a
  compact latent `z`. (Stage 1 is data preprocessing / tokenization.)
- **Stage 3 — LDM**: a DiT is trained with flow matching (`transport.py`) to
  generate latents conditioned on perturbation labels. An EMA copy of the DiT is
  maintained for sampling.
- **Generation**: latents are sampled with an ODE solver and classifier-free
  guidance, then decoded through the (frozen) VAE.

## Module layout

| File | Purpose |
| --- | --- |
| `build.py` | Hyperparameter dataclasses (`VAEHParams`, `LDMHParams`) and module builders. |
| `models.py` | Lightning modules: `VAE`, `LatentDiffusion`, `VAEScvi`, and `BaseModel`. |
| `vae.py` | `TransformerVAE` assembling encoder, decoder, NB head, input layer. |
| `nnets.py` | Encoder/Decoder and `DiT` network definitions. |
| `layers.py` | Attention blocks, MLPs, DiT layers, count projections. |
| `stochastic_layers.py` | Distribution heads (Negative Binomial, Gaussian). |
| `distributions.py` | Log-likelihood helpers (`log_nb_positive`, `log_gaussian`). |
| `transport.py` | Flow-matching / diffusion transport, ODE/SDE samplers. |
| `sde_path.py` | Interpolant path plans (Linear/GVP/VP). |
| `encoder.py` | `VocabularyEncoderSimplified` — gene/label ↔ index mapping. |
| `data.py` | `PerturbseqDataModule` and cell tokenization. |
| `evaluations.py` | MMD kernels and Wasserstein distances for evaluation. |
| `optimizers.py` | `AdamWLegacy` optimizer. |
| `runtime.py`  | LR schedule (`wsd_schedule`) and AnnData output helpers. |
| `constants.py` | `ModelEnum`, `LossEnum` batch/loss keys. |
| `train_vae.py` | Stage-2 VAE training entrypoint. |
| `train_ldm.py` | Stage-3 LDM training entrypoint (frozen VAE + DiT). |
| `generate.py` | Sampling / generation entrypoint with CFG. |
| `run_real.py` | Single-process real-data orchestrator (convert → VAE → LDM → generation) using codebase defaults. |
| `smoke_synthetic.py` | Fast end-to-end smoke test on synthetic data (VAE + LDM + generation). |

## Data

The training scripts expect a data directory containing:

- `train_asSTATE_hvg.h5ad`, `test_asSTATE_hvg.h5ad` — HVG-subset AnnData.
- `replogle_train.json` — gene/label vocabulary metadata.
- `replogle_log_size_factor_munew.pkl`, `replogle_log_size_factor_sdnew.pkl` —
  per-condition log size-factor statistics (mean / std) for size-factor sampling.

Replogle uses **joint** `(cell_line, gene)` conditioning. The DiT condition
embedding sizes are derived from the metadata categories at train time
(`PerturbseqDataModule.class_vocab_sizes`) and persisted to `hparams.json`, so
generation rebuilds the identical architecture — no hardcoded vocab sizes.

## Usage

Run from the `transportability/` directory. The scripts are modules, so invoke
them with `python -m`:

### 1. Prepare data (stage 1)

Stage 1 is offline preprocessing that produces the input artifacts consumed by
all later stages (see [Data](#data) above):

- HVG-subset, `asSTATE`-formatted AnnData (`train_asSTATE_hvg.h5ad`,
  `test_asSTATE_hvg.h5ad`).
- the gene/label vocabulary metadata (`replogle_train.json`).
- per-condition log size-factor statistics
  (`replogle_log_size_factor_munew.pkl`, `replogle_log_size_factor_sdnew.pkl`).

To feed scLDM, adapt that processed AnnData into the `asSTATE` shape scLDM's
`PerturbseqDataModule` expects (raw counts in `.X`; `cell_line` + `gene`
condition columns; the dataset's complete gene set), then emit `replogle_train.json`
(`{"genes": [...], "labels": {"cell_line": [...], "gene": [...]}}`) and the two
log-size-factor pkls (per joint `cell_line_gene` class). At train/generation
time, cells are tokenized on the fly by `data.tokenize_cells`; point
`--data-dir` at the directory holding these artifacts.

### 2. Train the VAE (stage 2)

```bash
python -m models.scldm.train_vae \
  --data-dir /path/to/replogle \
  --out-dir  /path/to/out/vae \
  --num-epochs 100 --batch-size 128 --precision bf16-mixed
```

Produces `vae.ckpt`. The data dir must contain a `val_asSTATE_hvg.h5ad` (override
with `--val-h5ad`): validation runs every `--val-check-every-n-epochs` epochs
(default 10) and the saved `vae.ckpt` is the best checkpoint by `val_loss`.

### 3. Train the LDM (stage 3)

```bash
python -m models.scldm.train_ldm \
  --data-dir /path/to/replogle \
  --vae-ckpt /path/to/out/vae/vae.ckpt \
  --out-dir  /path/to/out/ldm \
  --num-epochs 100 --batch-size 128 \
  --condition-strategy joint --precision bf16-mixed
```

The VAE is loaded frozen (`strict=True`) and the DiT is trained on top. The
`condition_strategy` is written to a sibling `hparams.json` for safe reload.
Validation uses `val_asSTATE_hvg.h5ad` (override with `--val-h5ad`) every
`--val-check-every-n-epochs` epochs (default 10); the saved `ldm.ckpt` is the
best checkpoint by `val_loss`.

### 4. Generate perturbations

```bash
python -m models.scldm.generate \
  --data-dir /path/to/replogle \
  --ldm-ckpt /path/to/out/ldm/ldm.ckpt \
  --out-dir  /path/to/out/gen \
  --omega 5 --timesteps 50
```

`--omega` is the classifier-free guidance weight (applied jointly to
`cell_line` and `gene`). The condition strategy is read from the checkpoint's
`hparams.json`; passing a disagreeing `--condition-strategy` raises an error.

### End-to-end on real data (single process)

`run_real.py` runs the whole pipeline — convert raw AnnData → `asSTATE` inputs,
train the VAE (stage 2), train the LDM (stage 3), then generate — in one process,
using the **codebase / paper defaults with no overrides** (`VAEHParams()` /
`LDMHParams()` == `vae_base.yaml` / `ldm_base.yaml`; epoch-driven `max_steps` via
`compute_training_steps`; `timesteps=50`; `omega=1.0`). It is the real-data
counterpart of `smoke_synthetic.py` and computes no reconstruction metric.

Call `run_scldm(...)` with raw train/val/test AnnData objects plus the column
mapping and hyperparameters:

```python
import anndata as ad
from models.scldm.run_real import run_scldm

result = run_scldm(
    train_adata=ad.read_h5ad("/path/to/train.h5ad"),
    val_adata=ad.read_h5ad("/path/to/val.h5ad"),
    test_adata=ad.read_h5ad("/path/to/test.h5ad"),
    out_dir="/path/to/out",
    perturbation_column="perturbation",
    context_key="cell_type",
    control_label="control",
)
```

- You pass **raw** AnnData objects; the function writes the `asSTATE` inputs to
  `<out_dir>/asstate/`. The condition vocabulary includes train, validation, and
  test labels, while library-size statistics are fitted on train + validation only.
- Set `perturbation_column` / `context_key` / `control_label` to match your
  AnnData `obs`; they map onto scLDM's `gene` / `cell_line` condition keys.
- `val_adata` mirrors the upstream training loop: validation runs every
  `val_check_every_n_epochs` epochs (upstream default 10) and the best checkpoint
  is selected by `val_loss` for both the VAE and the LDM.
- `run_scldm` returns a dict with the written artifact paths, the VAE/LDM
  checkpoints, the generated AnnData, its path, and the conditional/unconditional
  cell counts. Artifacts written under `out_dir`: `vae.ckpt`, `ldm.ckpt` +
  `hparams.json`, and `generated_<seed>.h5ad`. Training requires CUDA
  (`flex_attention` backward is GPU-only).

### Smoke test (synthetic data)

For a fast end-to-end sanity check that needs no real data, run the synthetic
smoke test. It generates a tiny perturbation dataset, writes the `asSTATE`
inputs, and exercises the full pipeline: VAE reconstruction (stage 2), then the
LDM build → `ldm.ckpt` + `hparams.json` → reload via `generate.py` →
classifier-free-guidance generation (stage 3 + generation).

```bash
python -m models.scldm.smoke_synthetic --out-dir /tmp/scldm_smoke
```

- The transformer attention uses `flex_attention`, whose backward is GPU-only.
  This smoke test currently requires CUDA and will exit if no GPU is available.
- Useful knobs: `--n-cells`, `--n-genes` (>= 200), `--timesteps` (ODE sampling
  steps), `--omega` (CFG weight). This validates the data-driven
  `class_vocab_sizes` round-trip (datamodule → `hparams.json` → generation).

## Notes

- **Checkpoint loading**: trusted, locally produced Lightning checkpoints are
  loaded explicitly with `weights_only=False` where direct `torch.load` calls are used.
- **`condition_strategy` is constructor-only** state (it changes
  `DiT.forward_with_cfg` and joint size-factor sampling) and is *not* in the
  `state_dict`, hence the `hparams.json` guard in `generate.py`. The DiT
  `class_vocab_sizes` are constructor-only in the same way (they size the
  condition embeddings), so they are persisted to `hparams.json` too and read
  back at generation time to rebuild the identical architecture.
