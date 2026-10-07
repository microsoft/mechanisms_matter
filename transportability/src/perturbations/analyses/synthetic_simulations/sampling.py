"""Shared synthetic-parameter sampling utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PARAM_RANGES = {
    "G": {"type": "log2_seq", "min": 1024, "max": 8192},
    "N0": {"type": "log2_seq", "min": 128, "max": 1024},
    "Nk": {"type": "log2_seq", "min": 128, "max": 512},
    "P": {"type": "int", "min": 20, "max": 50},
    "p_effect": {"type": "float", "min": 0.001, "max": 0.1},
    "effect_factor": {"type": "float", "min": 1.2, "max": 5.0},
    "B": {"type": "float", "min": 0.0, "max": 2.0},
    "mu_l": {"type": "float", "min": 0.2, "max": 5.0},
}
DEMO_PARAM_RANGES = {
    **PARAM_RANGES,
    "G": {"type": "fixed", "value": 64},
    "N0": {"type": "fixed", "value": 64},
    "Nk": {"type": "fixed", "value": 32},
    "P": {"type": "fixed", "value": 8},
}
CONTROL_PARAMS_PATH = "results/synthetic_simulations/parameter_estimation/control_fitted_params.csv"
PERTURBED_PARAMS_PATH = (
    "results/synthetic_simulations/parameter_estimation/perturbed_fitted_params.csv"
)
ALL_PARAMS_PATH = "results/synthetic_simulations/parameter_estimation/all_fitted_params.csv"


def make_demo_parameter_inputs() -> dict[str, np.ndarray]:
    """Generate small illustrative inputs without fitted data or external files."""
    n_genes = DEMO_PARAM_RANGES["G"]["value"]
    control_mu = np.linspace(0.5, 5.0, n_genes)
    return {
        "control_mu": control_mu,
        "pert_mu": control_mu * 1.5,
        "all_theta": np.full(n_genes, 5.0),
        "gene_names": np.asarray([f"gene{i}" for i in range(n_genes)]),
    }


def sample_parameters(
    param_ranges: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample one synthetic parameter set from configured ranges."""
    params: dict[str, Any] = {}
    for param, range_info in param_ranges.items():
        if range_info["type"] == "int":
            params[param] = int(rng.integers(range_info["min"], range_info["max"] + 1))
        elif range_info["type"] == "float":
            params[param] = float(rng.uniform(range_info["min"], range_info["max"]))
        elif range_info["type"] == "log2_seq":
            log_min = int(np.log2(range_info["min"]))
            log_max = int(np.log2(range_info["max"]))
            params[param] = int(2 ** float(rng.uniform(log_min, log_max)))
        elif range_info["type"] == "fixed":
            params[param] = range_info["value"]
        else:
            raise ValueError(f"Unsupported synthetic parameter type: {range_info['type']}")
    return params


def load_parameter_estimation_inputs() -> dict[str, np.ndarray]:
    """Load fitted parameter-estimation arrays used by synthetic generators."""
    control_params_df = pd.read_csv(CONTROL_PARAMS_PATH, index_col=0)
    perturbed_params_df = pd.read_csv(PERTURBED_PARAMS_PATH, index_col=0)
    all_params_df = pd.read_csv(ALL_PARAMS_PATH, index_col=0)
    return {
        "control_mu": control_params_df["mu"].to_numpy(),
        "pert_mu": perturbed_params_df["mu"].to_numpy(),
        "all_theta": all_params_df["n"].to_numpy(),
        "gene_names": control_params_df.index.to_numpy(dtype=str),
    }
