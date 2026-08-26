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

[CausalDGP](transportability/src/perturbations/data/dgp/causalDGP.py) is the proposed data-generating process for realistic synthetic single-cell perturbation data. It represents gene expression with stochastic causal dynamics,

$$
d\mathbf{x} = (A\mathbf{x} + \mathbf{b} + \mathbf{c}_q)\,dt + \sqrt{2}\,d\mathbf{W},
$$

where $A$ is a sparse gene-regulatory network, $\mathbf{b}$ is the baseline state, and $\mathbf{c}_q$ encodes perturbation $q$. The simulator generates two cellular contexts and can vary the regulatory network $A$, baseline state $\mathbf{b}$, both, or neither through `--diversity_type`. This makes it possible to test transportability under explicit, controlled changes to the underlying causal mechanism.

Latent steady-state expression is converted into sparse count data using gene-specific dispersion estimates fitted from Norman19. The returned `AnnData` contains raw counts in `.X`, normalized log-expression in `.layers["normalized_log1p"]`, perturbation and context labels in `.obs`, and the affected-gene masks for each context. Use the synthetic sweep under [Example Runs](#example-runs) to generate and evaluate CausalDGP datasets.

### Simulator Validation

[The simulator-validation analysis](transportability/src/perturbations/analyses/simulator_validation/generate_statistics.py) compares real and synthetic data using gene-wise and cell-wise marginal statistics, gene-pair correlations, and TRADE perturbation-effect statistics. It reports medians with bootstrap confidence intervals, either pooled or separately by context.

After completing the setup and changing to `transportability/src/perturbations/`, generate a CausalDGP validation summary with:

```bash
uv run python -m perturbations.analyses.simulator_validation.generate_statistics \
  --source synthetic \
  --name causalDGP \
  --G 128 \
  --P 128 \
  --diversity-type both \
  --output-dir results/simulator_validation
```

For synthetic data, the analysis defaults to reporting each `cell_line` context separately. It writes `causalDGP_validation_summary.csv` and `causalDGP_perturbation_effect.csv` to `results/simulator_validation/`.

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

## Metrics

The `perturbations` package provides evaluation metrics for perturbation models across three categories: perturbation effect, reconstruction, and gene selection. The [proposed Vendi score](transportability/src/perturbations/metrics/reconstruction/vendi_score.py) is used in two forms: a cell-level Vendi score for distributional reconstruction and a pseudobulk Vendi score for perturbation effects.

The [Vendi sensitivity analysis](transportability/src/perturbations/analyses/simulator_validation/vendi_sensitivity.py) evaluates how stable both forms of the score are under cell subsampling and injected dropout or Gaussian noise.

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
