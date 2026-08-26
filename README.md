# Mechanisms Matter: Transportability of Cellular Perturbation Effects

[![Code analysis](https://github.com/microsoft/mechanisms_matter/actions/workflows/code-analysis.yml/badge.svg)](https://github.com/microsoft/mechanisms_matter/actions/workflows/code-analysis.yml)

This repository contains the code accompanying our paper [Mechanisms Matter: Transportability of Cellular Perturbation Effects](https://www.biorxiv.org/content/10.64898/2026.05.08.723625v2), accepted at the 2026 Workshop on Generative and Agentic AI for Biology (ICML 2026). The project studies when perturbation effects can be transported across biological contexts using causal simulations, real Perturb-seq datasets, simple baselines, deep learning models, and diversity-aware evaluation metrics.

To cite the paper:

```bibtex
@inproceedings{
  qi2026mechanisms,
  title={Mechanisms Matter: Transportability of Cellular Perturbation Effects},
  author={Shi-ang Qi and Paidamoyo Chapfuwa},
  booktitle={2026 Workshop on Generative and Agentic AI for Biology (ICML 2026)},
  year={2026},
  doi={10.64898/2026.05.08.723625},
  url={https://www.biorxiv.org/content/10.64898/2026.05.08.723625v2}
}
```

- **Keywords**: Cellular Perturbation, Transportability, Causal Inference
- **License**: MIT

## Model

![Transportability framework](transportability_model-v2.png)

## Data

### CausalDGP

[CausalDGP](transportability/src/perturbations/data/dgp/causalDGP.py) is the proposed data-generating process for realistic *semi-synthetic* single-cell perturbation data. It models gene expression as an SDE,

$$
d\mathbf{x} = (A\mathbf{x} + B + \Gamma_q)\,dt + \sqrt{2}\,d\mathbf{W}
$$

where $A$ is a sparse gene-regulatory network, $B$ is the baseline state, $\Gamma_q$ is the perturbation effect, and $q$ is the perturbation. The simulator generates two cellular contexts and can vary $A$, $B$, both, or neither through `--diversity_type` to test transportability under controlled causal changes. See [Example Runs](#example-runs) to generate and evaluate CausalDGP datasets.

### Real Datasets

The following Perturb-seq datasets are used for real-data experiments:

| Dataset | Cell Lines | Source |
|---------|-----------|--------|
| Norman19 | K562 | [Norman et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/) |
| Replogle22 | K562, RPE1 | [Replogle et al. 2022](https://www.sciencedirect.com/science/article/pii/S0092867422005979) |
| Nadig25 | Jurkat, HepG2 | [Nadig et al. 2025](https://www.nature.com/articles/s41588-025-02169-3) |
| Zhu25 | Primary human CD4+ T cells across 4 donors | [Zhu et al. 2025](https://doi.org/10.64898/2025.12.23.696273) |

See the per-dataset README files under [`transportability/src/perturbations/data/`](transportability/src/perturbations/data/) for download and preprocessing instructions.

### Simulator Validation

[The simulator-validation analysis](transportability/src/perturbations/analyses/simulator_validation/generate_statistics.py) compares real and synthetic data using gene-wise and cell-wise marginal statistics, gene-pair correlations, and TRADE ([Nadig et al. 2025](https://www.nature.com/articles/s41588-025-02169-3)) perturbation-effect statistics.

## Context-Aware Splitting

![Context-aware splitting](context_splitting.png)

The [`ContextSplitter`](transportability/src/perturbations/analyses/context.py) partitions cells into train, validation, and test sets under two strategies that share a single seeded plan per trial for quantifying the cross-context generalization gap.

## Metrics

The `perturbations` package provides evaluation metrics for perturbation models across three categories: perturbation effect, reconstruction, and gene selection. The [proposed Vendi score](transportability/src/perturbations/metrics/reconstruction/vendi_score.py) is used in two forms: a cell-level Vendi score for distributional reconstruction and a pseudobulk Vendi score for perturbation effects.

The [Vendi sensitivity analysis](transportability/src/perturbations/analyses/simulator_validation/vendi_sensitivity.py) evaluates how stable both forms of the score are under cell subsampling and injected dropout or Gaussian noise.

## Prerequisites

- [Python ≥ 3.10](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) for dependency management

Install the `perturbations` package and its dependencies:

```bash
cd transportability
uv sync
```

## Example Runs

The analysis modules use paths relative to the package source directory. After setup, run the examples from that directory:

```bash
cd src/perturbations
```

Run a small synthetic sweep:

```bash
uv run python -m analyses.synthetic_simulations.random_sweep \
  --n_trials 2 \
  --dataset causalDGP \
  --split_strategy in-context \
  --diversity_type A
```

Run real-data experiments:

```bash
uv run python -m analyses.real_experiments.run \
  --dataset_name norman19 \
  --dataset_path data/norman19/norman19_processed.h5ad \
  --n_trials 10

uv run python -m analyses.real_experiments.run \
  --dataset_name replogle22 \
  --dataset_variant RPE1 \
  --split_strategy cross-context \
  --n_trials 10
```

For Replogle22, run one subset per job with `--dataset_variant RPE1`, `Jurkat`, or `HepG2`.

## Repository Layout

The main package is under `transportability/src/perturbations/`:

- `data/`: dataset download, preprocessing, and synthetic data generation
- `models/`: model wrappers and baselines
- `metrics/`: reconstruction, perturbation-effect, gene-selection, PDS, and diversity metrics
- `analyses/`: experiment runners and plotting scripts
- `results/`: generated experiment outputs

## Direct intended uses

This code is shared for research purposes to reproduce and extend the experiments in the paper. It is not intended for clinical use.

## Out-of-scope uses

This is a research codebase and should not be used in any clinical or production setting.

## Risks and limitations

Transportability conclusions are conditioned on the causal assumptions and datasets described in the paper. Results may not generalise to cell types, perturbations, or experimental platforms outside the evaluated conditions.

## License and Usage Notices

The code in this repository is provided for research use only. It is not intended for use in clinical decision-making or for any other clinical use, and performance for clinical use has not been established. You bear sole responsibility for any use of this code, including incorporation into any product intended for clinical use.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
