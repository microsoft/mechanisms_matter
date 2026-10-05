# Reproducible environments

Use Python 3.11 and uv 0.11.8 as the reference setup. Commit each `pyproject.toml`
with its adjacent `uv.lock`. The lockfiles record exact package versions, artifact
hashes, platform markers, and the Geneformer Git revision. Runtime packages come
from public PyPI; Geneformer PyTorch comes from the explicit public PyTorch indexes.

| Environment | Manifest and lock location | Purpose |
| --- | --- | --- |
| Main package | `transportability/` | Simulators, metrics, preprocessing, model dependencies, and tests |
| Notebooks | Repository root | Editable main package plus notebook tools, STATE and GEARS dependencies |
| Geneformer | `transportability/geneformer_env/` | Isolated embedding extraction with Transformers 4.x |

The main package owns the numerical dependency constraints: NumPy 2.2.6,
pandas 2.2.3, SciPy 1.15.2, scikit-learn 1.6.1, Scanpy 1.11.5, scvi-tools 1.3.3,
PyTorch 2.14.0, and Transformers 5.16.1. The notebook environment consumes those
constraints through the local `perturbations` package. Geneformer uses PyTorch
2.14.0, Transformers 4.57.6, datasets 2.21.0, and a fixed Geneformer source commit.
Its numerical stack is independently locked because it is a separate stage.

## Installation

For the main package, from the repository root:

```bash
cd transportability
uv --no-config sync --locked --python 3.11
uv run --locked python -c "import perturbations, scanpy, scvi, torch"
uv run --locked pytest -q
```

For notebooks, run `uv --no-config sync --locked --python 3.11` from the repository
root. For Geneformer, run this from the repository root:

```bash
cd transportability/geneformer_env
GIT_LFS_SKIP_SMUDGE=1 uv --no-config sync --locked --python 3.11
uv run --locked python -c "import geneformer, transformers, torch; print(torch.__version__, torch.version.cuda)"
```

`--no-config` avoids inheriting machine-specific uv configuration. `--locked`
checks manifest/lock agreement and prevents implicit dependency updates; see
[uv's locking documentation](https://docs.astral.sh/uv/concepts/projects/sync/).
Use `uv run --locked` for experiment commands too. A plain `pip install -e .`
does not use `uv.lock` and is not the reference environment.

Each project creates its own `.venv` by default. Do not activate Geneformer while
running main-package experiments. Exchange embeddings via `.h5ad` files with
`obsm["X_geneformer"]`, as described in the [Geneformer guide](../transportability/geneformer_env/README.md).

## Hardware and external artifacts

Linux x86_64 with Python 3.11 is the reference platform. Other supported Python
versions and operating systems can resolve different artifacts through lockfile
markers; they need separate runtime validation.

The main Linux PyTorch package includes CUDA libraries. Baseline and metric tests
can run on CPU; GPU model training needs a compatible NVIDIA driver. Geneformer
selects the CUDA 13.0 PyTorch index on Linux and the CPU index elsewhere. A
successful CPU import does not verify GPU execution. Check `torch.cuda.is_available()`
inside a GPU allocation before extraction or training. Package installation can
download several GB of CUDA libraries, so allow adequate disk space.

Data, model weights, and Geneformer tokenizer dictionaries are external artifacts,
separate from these package locks. `GIT_LFS_SKIP_SMUDGE=1` installs Geneformer code
without downloading weights. Fetch assets using the Geneformer guide before
extraction.

CPA, GEARS, STATE, and scLDM wrappers under `perturbations/models/` are excluded
from Git. Upstream package installation does not provide these repository-specific
wrappers. Full paper reproduction remains dependent on releasing or otherwise
supplying those implementations. Included baselines, simulators, and metrics can
be used independently; select included models explicitly in experiment commands.

## Alliance HPC

Install and run Python only inside a Slurm job or allocation. Verify your account
and available modules first:

```bash
sacctmgr show assoc user="$USER" format=Account -Pn
module spider python
module spider python/3.11.5
diskusage_report
```

`StdEnv/2023` and `python/3.11.5` were verified on Vulcan for this setup. Request
an allocation using your verified account, then enter its compute node:

```bash
salloc --account=<your-account> --cpus-per-task=2 --mem=16G --time=01:00:00
srun --pty bash
module --force purge
module load StdEnv/2023 python/3.11.5
export UV_PYTHON_DOWNLOADS=never
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venvs/mechanisms-matter-main"
export UV_CACHE_DIR="$SCRATCH/uv_cache"
export HF_HOME="$SCRATCH/hf_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
export PYTHONNOUSERSITE=1
cd <repository-path>/transportability
uv --no-config sync --locked --python "$(command -v python)"
```

This repository already uses uv. For an Alliance wheelhouse installation instead,
export pinned requirements from the lock, create a scratch virtualenv with
`virtualenv --no-download`, inspect availability with `avail_wheels`, and install
with `pip install --no-index`. Missing exact versions must be staged or obtained
through the cluster proxy; substitutions require fresh validation. A wheelhouse
environment is not automatically identical to the public PyPI artifact lock.

Give the notebook and Geneformer projects distinct `UV_PROJECT_ENVIRONMENT`
paths. Reload the same modules when reusing an environment. Keep caches, job logs,
temporary files, datasets, and experiment outputs on `$SCRATCH` or
`$SLURM_TMPDIR`; preserve cleaned results in project storage. The example scripts
use relative paths, so stage an experiment working directory on scratch before
running them on HPC.

## Updating and checking locks

Before changing dependencies, check all three projects:

```bash
uv --no-config lock --project . --check --offline
uv --no-config lock --project transportability --check --offline
uv --no-config lock --project transportability/geneformer_env --check --offline
```

After an intentional manifest edit, run `uv --no-config lock` in the affected
project. If the main package changes, also refresh the root notebook lock. Review
the dependency diff, then validate from a fresh environment with `uv sync --locked`,
`uv pip check`, import checks, and the existing tests before committing. CI checks
all three locks and installs the main environment with `--locked` on Python 3.11.

## Reference validation

On Vulcan on 2026-10-05, Slurm CPU jobs using Python 3.11.5 and uv 0.11.8 passed
fresh installs, dependency consistency checks, and runtime imports for all three
environments. All locks passed offline checks with an empty cache. Geneformer
imported with CUDA-enabled PyTorch, exposed the required Transformers API, and
passed the extraction CLI help check.

A source copy containing only tracked Python files passed 158 tests with one
expected optional STATE module skip. The full local workspace passed 164 tests.
GPU execution, model asset downloads, and full dataset experiments were not part
of this environment validation.
