"""
Hydra-free runtime helpers for scLDM.

Includes:
- wsd_schedule: warmup/hold/decay learning rate schedule
- process_generation_output: converts LDM predict_step outputs into AnnData
- create_anndata_from_inference_output: converts VAE inference outputs into AnnData
"""

# pyright: reportUnknownMemberType=false
from __future__ import annotations

import math
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .constants import ModelEnum
from .logger import logger


def wsd_schedule(
    num_training_steps: int,
    final_lr_factor: float = 0.1,
    num_warmup_steps: int = 1000,
    init_div_factor: float = 100,
    fract_decay: float = 0.1,
    decay_type: str = "cosine",
) -> Any:
    """
    Warmup, hold, and decay schedule.

    Args:
        num_training_steps: total number of training iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        num_warmup_steps: fraction of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
        decay_type: type of decay to apply after the hold phase; one of "cosine" or "sqrt"
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * num_training_steps)
    n_hold = num_training_steps - n_anneal_steps

    def schedule(step: int) -> float:
        if step < num_warmup_steps:
            return (step / num_warmup_steps) + (1 - step / num_warmup_steps) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < num_training_steps:
            if decay_type == "cosine":
                # Implement cosine decay from warmup to end
                decay_progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                return final_lr_factor + (1 - final_lr_factor) * 0.5 * (
                    1 + math.cos(math.pi * decay_progress)
                )
            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )
            else:
                raise ValueError(f"decay type {decay_type} is not in ['cosine','sqrt']")
        else:
            return final_lr_factor

    return schedule


def process_generation_output(
    output: list[dict[str, torch.Tensor]],
    datamodule: Any,
) -> ad.AnnData:
    """Process the generation output from the model and convert it into an AnnData object."""
    logger.info("Processing generation output")
    # counts_true_sparse = sparse.vstack([sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}"].numpy()) for o in output])
    counts_generated_unconditional_sparse = cast(
        sparse.csr_matrix,
        sparse.vstack(
            [
                sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}_generated_unconditional"].numpy())
                for o in output
            ]
        ),
    )
    counts_generated_conditional_sparse = cast(
        sparse.csr_matrix,
        sparse.vstack(
            [
                sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}_generated_conditional"].numpy())
                for o in output
            ]
        ),
    )
    z_generated_unconditional = np.vstack([o["z_generated_unconditional"].numpy() for o in output])
    z_generated_conditional = np.vstack([o["z_generated_conditional"].numpy() for o in output])

    genes = output[0][ModelEnum.GENES.value][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)

    # Only decode labels that are present in the output; skip missing ones
    available_keys = cast(
        "set[str]",
        set.intersection(*[set(o.keys()) for o in output]) if output else set(),
    )
    desired_keys = set(datamodule.vocabulary_encoder.labels.keys())
    present_label_keys = sorted(desired_keys & available_keys)
    missing_label_keys = sorted(desired_keys - available_keys)
    if missing_label_keys:
        logger.info(
            f"[generation_output] Skipping missing label columns in outputs: {missing_label_keys}"
        )

    # Stack label tensors/arrays robustly, then decode
    obs: dict[str, np.ndarray] = {}
    for k in present_label_keys:
        parts: list[np.ndarray] = []
        for o in output:
            v = o[k]
            if torch.is_tensor(v):
                parts.append(v.detach().cpu().numpy())
            else:
                parts.append(np.asarray(v))
        stacked = np.concatenate(parts, axis=0)
        obs[k] = datamodule.vocabulary_encoder.decode_metadata(stacked, k)

    del output

    n_cells = cast("tuple[int, int]", counts_generated_unconditional_sparse.shape)[0]

    obs_generated_unconditional = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))
    obs_generated_conditional = pd.DataFrame(obs, index=np.arange(n_cells, 2 * n_cells).astype(str))

    obs_generated_unconditional["dataset"] = "generated_unconditional"
    obs_generated_conditional["dataset"] = "generated_conditional"

    X_combined = cast(
        sparse.csr_matrix,
        sparse.vstack([counts_generated_unconditional_sparse, counts_generated_conditional_sparse]),
    )
    z_combined = np.vstack([z_generated_unconditional, z_generated_conditional])

    obs_combined = pd.concat([obs_generated_unconditional, obs_generated_conditional], axis=0)
    adata = ad.AnnData(X=X_combined, obs=obs_combined, obsm={"z": z_combined})  # pyright: ignore[reportArgumentType]
    adata.var_names = var_names
    return adata


def create_anndata_from_inference_output(
    output: dict[str, torch.Tensor],
    datamodule: Any,
) -> ad.AnnData:
    """Create an AnnData object from the inference output of the model, decoding the latent variables, counts, and metadata."""
    obsm: dict[str, np.ndarray] = {}
    if "z" in output:
        obsm["z"] = output["z"].numpy()
    if "z_mean_flat" in output:
        obsm["z_mean_flat"] = output["z_mean_flat"].numpy()
    if "z_sample_flat" in output:
        obsm["z_sample_flat"] = output["z_sample_flat"].numpy()
    if not obsm:
        raise KeyError(f"Missing latent keys in inference output. Keys: {sorted(output.keys())}")

    if "reconstructed_counts" in output:
        generated_counts = sparse.csr_matrix(output["reconstructed_counts"].numpy())
    elif ModelEnum.COUNTS.value in output:
        generated_counts = sparse.csr_matrix(output[ModelEnum.COUNTS.value].numpy())
    else:
        raise KeyError(f"Missing counts in inference output. Keys: {sorted(output.keys())}")

    if ModelEnum.GENES.value not in output:
        raise KeyError(f"Missing genes in inference output. Keys: {sorted(output.keys())}")
    genes = output[ModelEnum.GENES.value][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)
    if datamodule.vocabulary_encoder.labels is not None:
        obs = {
            k: datamodule.vocabulary_encoder.decode_metadata(output[k].numpy(), k)
            for k in datamodule.vocabulary_encoder.labels.keys()
        }
    else:
        obs = {}

    n_cells = cast("tuple[int, int]", generated_counts.shape)[0]
    obs = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))

    adata = ad.AnnData(
        X=generated_counts,
        obs=obs,
        obsm=obsm,  # pyright: ignore[reportArgumentType]
    )
    adata.var_names = var_names
    adata.layers["counts"] = generated_counts.copy()
    return adata
