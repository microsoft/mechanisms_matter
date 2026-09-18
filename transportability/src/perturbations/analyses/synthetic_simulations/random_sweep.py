"""Run random-parameter synthetic simulation sweeps and collect evaluation metrics."""

import argparse
import multiprocessing
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from perturbations.metrics.reconstruction.distance_util import (
    estimate_mmd_gamma,
)
from perturbations.metrics.reconstruction.vendi_score import (
    estimate_vendi_outer_sigma_squared,
    estimate_vendi_pseudobulk_sigma_squared,
    fit_vendi_pseudobulk_pca,
    vendi_score,
    vendi_score_pseudobulk,
)
from perturbations.models.linear import predict_linear_pca_baseline
from perturbations.models.optional import (
    ScviPerturbation,
    run_cpa,
    run_gears,
    run_scldm,
    run_state_gene,
)
from perturbations.util.anndata_util import fit_control_incremental_pca

from ...data.dgp import causalDGP, directDGP
from ...metrics.gene_selection.differential_expression_score import (
    de_table_to_deg_masks,
    scanpy_de_table,
)
from ..common import MODELS, NORM_LAYER_KEY, label_to_target_tokens
from ..context import (
    ContextSplitter,
    build_evaluation_contexts,
    filter_indices_to_context_values,
    indexer_from_labels,
)
from ..evaluator import evaluation, get_eval_perturbation_ids
from ..util import (
    build_perturbation_id_map,
    compute_means_by_perturbation,
    est_cost,
    intra_correlation,
    systematic_variation,
    true_degs_for_context,
)
from .sampling import (
    ALL_PARAMS_PATH,
    PARAM_RANGES,
    load_parameter_estimation_inputs,
    sample_parameters,
)

# Set OpenBLAS threads early if it was found to be helpful, otherwise optional
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

_GLOBAL = {}
_OUTPUT_DIR = "results/synthetic_simulations/random_sweep_results"


def get_data_stats(
    adata,
    P: int,
    trial_id_for_rng: int | None,
) -> dict[str, float]:
    """Compute dataset-level summary statistics for one synthetic AnnData object."""
    analysis_batch_size = max(1, min(1024, int(adata.n_obs)))
    counts_matrix = adata.X
    total_entries = int(adata.n_obs * adata.n_vars)
    if sparse.issparse(counts_matrix):
        total_nonzero = int(counts_matrix.nnz)
        library_sizes = np.asarray(counts_matrix.sum(axis=1)).ravel().astype(np.float64, copy=False)
    else:
        counts_dense = np.asarray(counts_matrix)
        total_nonzero = int(np.count_nonzero(counts_dense))
        library_sizes = counts_dense.sum(axis=1, dtype=np.float64)

    perturbation_ids = np.arange(P, dtype=np.int32)
    mu_obs = compute_means_by_perturbation(
        adata_view=adata,
        perturbation_ids=perturbation_ids,
        layer_key=NORM_LAYER_KEY,
        perturbation_key="perturbation_id",
    )
    mu_control = compute_means_by_perturbation(
        adata_view=adata,
        perturbation_ids=np.asarray([-1], dtype=np.int32),
        layer_key=NORM_LAYER_KEY,
        perturbation_key="perturbation_id",
    )[0]
    mu_pool = mu_obs.mean(axis=0).astype(np.float32, copy=False)
    vendi_random_state = 0 if trial_id_for_rng is None else int(trial_id_for_rng)
    vendi_pca_model = fit_control_incremental_pca(
        data_obj=adata,
        layer_key=NORM_LAYER_KEY,
        control_label="control",
        obs_key="perturbation",
        data_name="adata",
    )
    vendi_gamma = estimate_mmd_gamma(
        obs=adata,
        layer_obs=NORM_LAYER_KEY,
        control_label="control",
        seed=vendi_random_state,
        pca_model=vendi_pca_model,
    )
    vendi_outer_sigma_squared = estimate_vendi_outer_sigma_squared(
        ac=adata,
        gamma=vendi_gamma,
        pca_model=vendi_pca_model,
        layer_key=NORM_LAYER_KEY,
        control_label="control",
        random_state=vendi_random_state,
    )

    return {
        "sparsity": 1.0 - (total_nonzero / total_entries),
        "median_library_size": (
            float(np.nanmedian(library_sizes)) if library_sizes.size > 0 else np.nan
        ),
        "systematic_variation": systematic_variation(
            ptb_shifts=mu_obs - mu_control,
            avg_ptb_shift=mu_pool - mu_control,
        ),
        "intra_corr": intra_correlation(mu_obs - mu_control),
        "vendi_score": vendi_score(
            ac=adata,
            ac_batch_size=analysis_batch_size,
            n_pca_components=50,
            sample_size=2_000,
            random_state=vendi_random_state,
            layer_key=NORM_LAYER_KEY,
            control_label="control",
            gamma=vendi_gamma,
            pca_model=vendi_pca_model,
            outer_sigma_squared=vendi_outer_sigma_squared,
        ),
    }


def _run_scldm_prediction(
    *,
    adata: ad.AnnData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    context_key: str,
    pid: int,
    trial_seed: int,
) -> ad.AnnData:
    """Train scLDM and return normalized predictions aligned to the synthetic test split."""
    from perturbations.models.scldm_attention import configure_synthetic_attention

    configure_synthetic_attention()
    scldm_out = run_scldm(
        train_adata=adata[train_idx, :],
        val_adata=adata[val_idx, :],
        test_adata=adata[test_idx, :],
        out_dir=str(
            Path("results") / "synthetic_simulations" / "scldm" / str(pid) / f"trial_{trial_seed}"
        ),
        perturbation_column="perturbation",
        context_key=context_key,
        control_label="control",
        num_epochs=100,
        seed=trial_seed,
        resume=False,
        counts_layer=None,
        normalized_target_sum=1e4,
    )
    scldm_predictions = np.asarray(scldm_out["preds"], dtype=np.float32)
    prediction = adata[test_idx, :].copy()
    prediction.layers[NORM_LAYER_KEY] = scldm_predictions
    del scldm_out
    return prediction


def simulate_one_run(
    dataset_name: str,
    G: int,  # number of genes
    N0: int,  # number of control cells
    Nk: int,  # number of perturbed cells per perturbation
    P: int,  # number of perturbations
    p_effect: float,  # a threshold for fraction of genes affected per perturbation
    effect_factor: float,  # effect factor for affected genes, epsilon in the paper
    B: float,  # global perturbation bias factor, beta in the paper
    mu_l: float,  # mean of log library size
    diversity_type: str,  # Type of diversity to introduce for causalDGP dataset ("A", "b", "both", or "none")
    all_theta: np.ndarray,  # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    control_mu: np.ndarray,  # Control mu parameters, size of total number of genes in the real dataset (>= G)
    pert_mu: np.ndarray,  # Perturbed mu parameters, size of total number of genes in the real dataset (>= G)
    gene_names: np.ndarray,  # Gene names loaded from parameter estimation results
    pid: int,  # Process ID for this run, used for caching modelings (GEARS and CPA)
    trial_id_for_rng: int | None = None,  # Optional for seeding RNG per trial,
    normalize: bool = True,  # Whether to normalize the data
    split_strategy: str = "in-context",  # Data splitting strategy for evaluation
):
    """Simulate one synthetic experiment using a single in-memory AnnData object."""
    if dataset_name == "directDGP" and split_strategy == "cross-context":
        print(
            "Warning: 'cross-context' split strategy is not applicable to 'directDGP' dataset. "
            "Defaulting to 'in-context' split strategy for this dataset."
        )
        split_strategy = "in-context"

    if dataset_name == "directDGP":
        adata, affected_genes = directDGP(
            G=G,
            N0=N0,
            Nk=Nk,
            P=P,
            p_effect=p_effect,
            effect_factor=effect_factor,
            B=B,
            mu_l=mu_l,
            all_theta=all_theta,
            control_mu=control_mu,
            pert_mu=pert_mu,
            gene_names=gene_names,
            seed=trial_id_for_rng,
            normalize=normalize,
            normalized_layer_key=NORM_LAYER_KEY,
        )
    elif dataset_name == "causalDGP":
        adata, affected_genes = causalDGP(
            G=G,
            N0=N0,
            Nk=Nk,
            P=P,
            mu_l=1,
            all_theta=all_theta,
            gene_names=gene_names,
            mask_method="Erdos-Renyi",
            diversity_type=diversity_type,
            swap_fraction=0.5,
            seed=trial_id_for_rng,
            normalize=normalize,
            normalized_layer_key=NORM_LAYER_KEY,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    data_stats = get_data_stats(
        adata=adata,
        P=P,
        trial_id_for_rng=trial_id_for_rng,
    )

    splitter = ContextSplitter(
        adata=adata,
        split_strategy=split_strategy,
        test_context_values=[1] if dataset_name == "causalDGP" else None,
    )
    train_idx, val_idx, test_idx = splitter.split(seed=trial_id_for_rng)
    split_metadata = splitter.get_split_metadata()

    labels = adata.obs["perturbation"].to_numpy(copy=False)
    train_val_idx = np.concatenate([train_idx, val_idx])

    ad_test_only = adata[test_idx, :]
    test_perturbation_ids = get_eval_perturbation_ids(
        obs=ad_test_only,
        control_label="control",
        strict_match=False,
    )
    if test_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbations found in test split.")

    ad_train_val = adata[train_val_idx, :]
    gene_names = np.asarray(adata.var_names).astype(str)

    train_perturbation_ids = get_eval_perturbation_ids(
        obs=ad_train_val,
        control_label="control",
        strict_match=False,
    )
    if train_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbations found in train/val split.")

    mu_train = compute_means_by_perturbation(
        adata_view=ad_train_val,
        perturbation_ids=train_perturbation_ids,
        layer_key=NORM_LAYER_KEY,
    )
    control_id = np.asarray(["control"], dtype=object)
    mu_control_train = compute_means_by_perturbation(
        adata_view=ad_train_val,
        perturbation_ids=control_id,
        layer_key=NORM_LAYER_KEY,
    )[0]
    mu_pool_train = mu_train.mean(axis=0).astype(np.float32, copy=False)

    pert_label_to_id = build_perturbation_id_map(adata.obs)
    train_targets = [label_to_target_tokens(str(pid)) for pid in train_perturbation_ids]
    test_targets = [label_to_target_tokens(str(pid)) for pid in test_perturbation_ids]
    eval_contexts = build_evaluation_contexts(splitter)
    test_control_idx = test_idx[labels[test_idx] == "control"]
    train_perturbed_idx = train_idx[labels[train_idx] != "control"]

    eval_specs: list[dict[str, Any]] = []
    for eval_context in eval_contexts:
        bucket_test_idx = filter_indices_to_context_values(
            obs=adata.obs,
            indices=test_idx,
            context_axis=eval_context.axis,
            context_values=eval_context.values,
        )
        bucket_control_idx = filter_indices_to_context_values(
            obs=adata.obs,
            indices=test_control_idx,
            context_axis=eval_context.axis,
            context_values=eval_context.values,
        )
        bucket_train_context_idx = filter_indices_to_context_values(
            obs=adata.obs,
            indices=train_perturbed_idx,
            context_axis=eval_context.axis,
            context_values=eval_context.values,
        )
        if bucket_test_idx.size == 0 or bucket_control_idx.size == 0:
            continue
        if bucket_train_context_idx.size == 0:
            raise ValueError(
                "No non-control training cells found for context-specific baselines in "
                f"context {eval_context.value_label!r}."
            )

        bucket_test_eval = adata[bucket_test_idx, :].copy()
        bucket_perturbation_ids = get_eval_perturbation_ids(
            obs=bucket_test_eval,
            control_label="control",
            strict_match=False,
        )
        if bucket_perturbation_ids.size == 0:
            continue

        bucket_mu_obs = compute_means_by_perturbation(
            adata_view=bucket_test_eval,
            perturbation_ids=bucket_perturbation_ids,
            layer_key=NORM_LAYER_KEY,
        )
        bucket_mu_control = compute_means_by_perturbation(
            adata_view=bucket_test_eval,
            perturbation_ids=control_id,
            layer_key=NORM_LAYER_KEY,
        )[0]
        bucket_labels = bucket_test_eval.obs["perturbation"].to_numpy(copy=False)
        bucket_has_control = bool(np.any(np.asarray(bucket_labels) == "control"))
        bucket_counts = np.array(
            [np.sum(bucket_labels == pert_id) for pert_id in bucket_perturbation_ids],
            dtype=np.float64,
        )
        bucket_mu_pool = np.average(bucket_mu_obs, axis=0, weights=bucket_counts).astype(
            np.float32,
            copy=False,
        )
        bucket_train_context = adata[bucket_train_context_idx, :]
        bucket_train_context_perturbation_ids = get_eval_perturbation_ids(
            obs=bucket_train_context,
            control_label="control",
            strict_match=False,
        )
        bucket_mu_train_context = compute_means_by_perturbation(
            adata_view=bucket_train_context,
            perturbation_ids=bucket_train_context_perturbation_ids,
            layer_key=NORM_LAYER_KEY,
        )
        bucket_train_context_labels = bucket_train_context.obs["perturbation"].to_numpy(copy=False)
        bucket_train_context_counts = np.array(
            [
                np.sum(bucket_train_context_labels == pert_id)
                for pert_id in bucket_train_context_perturbation_ids
            ],
            dtype=np.float64,
        )
        bucket_mu_pool_train_context = np.average(
            bucket_mu_train_context,
            axis=0,
            weights=bucket_train_context_counts,
        ).astype(np.float32, copy=False)
        bucket_train_context_targets = [
            label_to_target_tokens(str(pid)) for pid in bucket_train_context_perturbation_ids
        ]
        bucket_test_targets = [label_to_target_tokens(str(pid)) for pid in bucket_perturbation_ids]
        bucket_de_table = scanpy_de_table(
            adata=bucket_test_eval,
            pert_col="perturbation",
            control_pert="control",
            key_added="test_de",
            layer=NORM_LAYER_KEY,
        )
        bucket_degs = de_table_to_deg_masks(
            de_table=bucket_de_table,
            gene_names=gene_names,
            perturbation_ids=bucket_perturbation_ids,
            fdr_threshold=0.05,
        )
        bucket_indexer = indexer_from_labels(
            all_labels=test_perturbation_ids,
            selected_labels=bucket_perturbation_ids,
        )
        bucket_mmd_pca_model = None
        if bucket_has_control:
            bucket_mmd_pca_model = fit_control_incremental_pca(
                data_obj=bucket_test_eval,
                layer_key=NORM_LAYER_KEY,
                control_label="control",
                obs_key="perturbation",
                data_name="bucket_test_eval",
            )
        if bucket_mmd_pca_model is None:
            raise ValueError("Vendi calibration requires observed control cells.")
        bucket_mmd_gamma = estimate_mmd_gamma(
            obs=bucket_test_eval,
            layer_obs=NORM_LAYER_KEY,
            control_label="control",
            pca_model=bucket_mmd_pca_model,
        )
        bucket_vendi_outer_sigma_squared = estimate_vendi_outer_sigma_squared(
            ac=bucket_test_eval,
            gamma=bucket_mmd_gamma,
            pca_model=bucket_mmd_pca_model,
            layer_key=NORM_LAYER_KEY,
            control_label="control",
        )
        bucket_vendi_pseudobulk_pca_model = None
        bucket_vendi_pseudobulk_sigma_squared = None
        if bucket_mu_obs.shape[0] > 1:
            bucket_vendi_pseudobulk_pca_model = fit_vendi_pseudobulk_pca(bucket_mu_obs)
            bucket_vendi_pseudobulk_sigma_squared = estimate_vendi_pseudobulk_sigma_squared(
                ac=bucket_test_eval,
                pca_model=bucket_vendi_pseudobulk_pca_model,
                layer_key=NORM_LAYER_KEY,
                control_label="control",
            )
        eval_specs.append(
            {
                "context_axis": eval_context.axis,
                "context_values": eval_context.value_label,
                "context_value_tuple": eval_context.values,
                "obs_eval": bucket_test_eval,
                "perturbation_ids": bucket_perturbation_ids,
                "mu_obs": bucket_mu_obs,
                "mu_control_obs": bucket_mu_control,
                "mu_pool_obs": bucket_mu_pool,
                "mu_train_context": bucket_mu_train_context,
                "mu_pool_train_context": bucket_mu_pool_train_context,
                "train_context_targets": bucket_train_context_targets,
                "test_targets": bucket_test_targets,
                "obs_DE_table": bucket_de_table,
                "obs_DEGs": bucket_degs,
                "mmd_gamma": bucket_mmd_gamma,
                "mmd_pca_model": bucket_mmd_pca_model,
                "vendi_outer_sigma_squared": bucket_vendi_outer_sigma_squared,
                "vendi_pseudobulk_pca_model": bucket_vendi_pseudobulk_pca_model,
                "vendi_pseudobulk_sigma_squared": bucket_vendi_pseudobulk_sigma_squared,
                "mu_pred_indexer": bucket_indexer,
                "true_DEGs": true_degs_for_context(
                    dataset_name=dataset_name,
                    affected_genes=affected_genes,
                    pert_label_to_id=pert_label_to_id,
                    perturbation_ids=bucket_perturbation_ids,
                    context_axis=eval_context.axis,
                    context_values=eval_context.values,
                ),
                # The evaluator reuses this value instead of recomputing it.
                # Cache the same pseudobulk metric and calibration used for predictions.
                "vendi_score_obs": vendi_score_pseudobulk(
                    bucket_mu_obs,
                    pca_model=bucket_vendi_pseudobulk_pca_model,
                    outer_sigma_squared=bucket_vendi_pseudobulk_sigma_squared,
                ),
            }
        )

    if not eval_specs:
        raise ValueError(
            "No evaluation contexts contained both held-out controls and non-control perturbations."
        )

    all_results = []
    for model in MODELS:
        start_time = time.time()
        mu_pred = None
        ad_test_pred = None
        if model == "Control":
            mu_pred = np.tile(mu_control_train, (test_perturbation_ids.size, 1))
        elif model == "Average":
            mu_pred = np.tile(mu_pool_train, (test_perturbation_ids.size, 1))
        elif model == "Context-Average":
            mu_pred = None
        elif model == "Context-linearPCA":
            mu_pred = None
        elif model == "linearPCA":
            mu_pred = predict_linear_pca_baseline(
                train_means=mu_train,
                train_target_genes=train_targets,
                test_target_genes=test_targets,
                gene_names=gene_names,
                fallback_mean=mu_pool_train,
                seed=trial_id_for_rng if trial_id_for_rng is not None else 42,
            )
        elif model == "scVI":
            scvi_model = ScviPerturbation(
                data=adata,
                counts_layer=None,
                perturbation_key="perturbation",
                context_key=split_metadata.context_axis,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=trial_id_for_rng if trial_id_for_rng is not None else 42,
            )
            ad_test_pred = scvi_model.run(
                n_latent=10,
                n_hidden=128,
                n_layers=3,
                gene_likelihood="zinb",
                dispersion="gene",
                max_epochs=800,
                batch_size=512,
                early_stopping=True,
                dataloader_num_workers=0,
                normalized_target_sum=1e4,
            )
        elif model == "GEARS":
            gears_out = run_gears(
                train_adata=adata[train_idx, :],
                valid_adata=adata[val_idx, :],
                test_adata=adata[test_idx, :],
                is_synthetic=True,
                dataset_name=f"{dataset_name}_trial_{trial_id_for_rng if trial_id_for_rng is not None else 0}",
                model_dir=str(
                    Path("results")
                    / "synthetic_simulations"
                    / "gears"
                    / str(pid)
                    / f"trial_{trial_id_for_rng if trial_id_for_rng is not None else 0}"
                ),
                perturbation_column="perturbation",
                control_label="control",
                expression_layer=NORM_LAYER_KEY,
                use_gene_ontology_graph=False,
                normalized_target_sum=1e4,
            )
            gears_pred = gears_out["preds"].detach().cpu().numpy().astype(np.float32, copy=False)
            ad_test_pred = adata[test_idx, :].copy()
            ad_test_pred.layers[NORM_LAYER_KEY] = gears_pred
        elif model == "CPA":
            cpa_out = run_cpa(
                train_adata=adata[train_idx, :],
                valid_adata=adata[val_idx, :],
                test_adata=adata[test_idx, :],
                add_controls=True,
                perturbation_column="perturbation",
                control_label="control",
                use_counts=False,
                epochs=100,
                dataset_name=f"{dataset_name}_trial_{trial_id_for_rng if trial_id_for_rng is not None else 0}",
                model_dir=str(
                    Path("results")
                    / "synthetic_simulations"
                    / "cpa"
                    / str(pid)
                    / f"trial_{trial_id_for_rng if trial_id_for_rng is not None else 0}"
                ),
                expression_layer=NORM_LAYER_KEY,
                covariate_keys=[split_metadata.context_axis],
                library_size=None,
            )
            cpa_pred = np.asarray(cpa_out["preds"], dtype=np.float32)
            ad_test_pred = adata[test_idx, :].copy()
            ad_test_pred.layers[NORM_LAYER_KEY] = cpa_pred
        elif model == "STATE":
            state_out = run_state_gene(
                train_adata=adata[train_idx, :],
                valid_adata=adata[val_idx, :],
                test_adata=adata[test_idx, :],
                context_key=split_metadata.context_axis,
                dataset_name=f"{dataset_name}_trial_{trial_id_for_rng if trial_id_for_rng is not None else 0}",
                model_dir=str(
                    Path("results")
                    / "synthetic_simulations"
                    / "state_gene"
                    / str(pid)
                    / f"trial_{trial_id_for_rng if trial_id_for_rng is not None else 0}"
                ),
                perturbation_column="perturbation",
                control_label="control",
                expression_layer=NORM_LAYER_KEY,
                epochs=100,
            )
            state_pred = np.asarray(state_out["preds"], dtype=np.float32)
            ad_test_pred = adata[test_idx, :].copy()
            ad_test_pred.layers[NORM_LAYER_KEY] = state_pred
        elif model == "scLDM":
            trial_seed = trial_id_for_rng if trial_id_for_rng is not None else 42
            ad_test_pred = _run_scldm_prediction(
                adata=adata,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                context_key=split_metadata.context_axis,
                pid=pid,
                trial_seed=trial_seed,
            )
        else:
            raise NotImplementedError(f"Model '{model}' is not implemented.")

        execution_time = time.time() - start_time
        for eval_spec in eval_specs:
            pred_for_eval = ad_test_pred
            if pred_for_eval is not None and eval_spec["context_axis"] is not None:
                pred_context_mask = np.isin(
                    pred_for_eval.obs[eval_spec["context_axis"]].to_numpy(copy=False),
                    eval_spec["context_value_tuple"],
                )
                if not np.all(pred_context_mask):
                    pred_for_eval = pred_for_eval[pred_context_mask, :].copy()

            if model == "Context-Average":
                mu_pred_for_eval = np.tile(
                    eval_spec["mu_pool_train_context"],
                    (eval_spec["perturbation_ids"].size, 1),
                )
            elif model == "Context-linearPCA":
                mu_pred_for_eval = predict_linear_pca_baseline(
                    train_means=eval_spec["mu_train_context"],
                    train_target_genes=eval_spec["train_context_targets"],
                    test_target_genes=eval_spec["test_targets"],
                    gene_names=gene_names,
                    fallback_mean=eval_spec["mu_pool_train_context"],
                    seed=trial_id_for_rng if trial_id_for_rng is not None else 42,
                )
            elif mu_pred is None:
                mu_pred_for_eval = None
            else:
                mu_pred_for_eval = mu_pred[eval_spec["mu_pred_indexer"], :]

            model_results = evaluation(
                pred=pred_for_eval,
                obs=eval_spec["obs_eval"],
                mu_pred=mu_pred_for_eval,
                mu_obs=eval_spec["mu_obs"],
                mu_control_obs=eval_spec["mu_control_obs"],
                mu_pool_obs=eval_spec["mu_pool_obs"],
                true_DEGs=eval_spec["true_DEGs"],
                obs_DE_table=eval_spec["obs_DE_table"],
                obs_DEGs=eval_spec["obs_DEGs"],
                mmd_gamma=eval_spec["mmd_gamma"],
                mmd_pca_model=eval_spec["mmd_pca_model"],
                vendi_outer_sigma_squared=eval_spec["vendi_outer_sigma_squared"],
                vendi_pseudobulk_pca_model=eval_spec["vendi_pseudobulk_pca_model"],
                vendi_pseudobulk_sigma_squared=eval_spec["vendi_pseudobulk_sigma_squared"],
                vendi_score_obs=eval_spec["vendi_score_obs"],
                perturbation_ids=eval_spec["perturbation_ids"],
                model=model,
                layer_name=NORM_LAYER_KEY,
                control_label="control",
            )

            model_results.update(
                {
                    "model": model,
                    "execution_time": execution_time,
                    "context_axis": eval_spec["context_axis"],
                    "context_values": eval_spec["context_values"],
                    **data_stats,
                }
            )
            all_results.append(model_results)

    return all_results


def order_tasks_for_pool(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Order trials from lighter to heavier estimated cost.

    Launching the largest in-memory synthetic trials first makes every worker grab
    a peak-memory job immediately, which can stall progress in multiprocessing runs.
    """
    return sorted(tasks, key=lambda task: est_cost(task["params_dict"]))


def init_worker(control_mu, all_theta, pert_mu, gene_names):
    """Initialize process-local global arrays for worker execution."""
    _GLOBAL["control_mu"] = control_mu
    _GLOBAL["all_theta"] = all_theta
    _GLOBAL["pert_mu"] = pert_mu
    _GLOBAL["gene_names"] = gene_names


# Revised _pool_worker to include timing (matches spirit of original)
def _pool_worker_timed(task_info_dict):
    dataset_name = task_info_dict["dataset_name"]
    trial_id = task_info_dict["trial_id"]
    params_dict = task_info_dict["params_dict"]
    split_strategy = task_info_dict["split_strategy"]
    diversity_type = task_info_dict["diversity_type"]
    pid = task_info_dict["pid"]
    control_mu_from_main = _GLOBAL["control_mu"]
    all_theta_from_main = _GLOBAL["all_theta"]
    pert_mu_from_main = _GLOBAL["pert_mu"]
    gene_names_from_main = _GLOBAL["gene_names"]

    # Add trial_id for RNG seeding within simulate_one_run.
    params_for_sim = params_dict.copy()  # Avoid modifying original params_dict
    params_for_sim["dataset_name"] = dataset_name
    params_for_sim["trial_id_for_rng"] = trial_id
    params_for_sim["split_strategy"] = split_strategy
    params_for_sim["diversity_type"] = diversity_type
    params_for_sim["pid"] = pid
    params_for_sim["control_mu"] = control_mu_from_main
    params_for_sim["all_theta"] = all_theta_from_main
    params_for_sim["pert_mu"] = pert_mu_from_main
    params_for_sim["gene_names"] = gene_names_from_main
    try:
        # Ensure all required keys used by simulate_one_run are present in params_for_sim.
        # G, N0, Nk, P, p_effect, effect_factor are expected from sample_parameters
        results_per_sim = simulate_one_run(**params_for_sim)

        # Prepare results: original sampled params + metrics + supporting info
        # `params_dict` is the original sampled params.
        final_results_per_sim = []
        for results_per_sim_model in results_per_sim:
            final_results_per_sim.append(
                {
                    **params_dict,
                    **results_per_sim_model,
                    "dataset": dataset_name,
                    "split_strategy": split_strategy,
                    "diversity_type": diversity_type,
                    "trial_id": trial_id,
                    "status": "success",
                }
            )
        return final_results_per_sim

    except Exception as e:
        traceback.print_exc()
        # Define metrics_error_keys locally for safety
        metrics_error_keys_local = {
            "pearson",
            "pearson_true_degs",
            "pearson_degs",
            "mae",
            "mae_true_degs",
            "mae_degs",
            "mse",
            "mse_true_degs",
            "mse_degs",
            "r2",
            "r2_true_degs",
            "r2_degs",
            "parametric_distance",
            "mmd_distance",
            "fid_distance",
            "des_recall",
            "des_precision",
            "des_jaccard",
            "vendi_score_pred",
            "vendi_score_obs",
            "pds_l1",
            "pds_l2",
            "pds_cosine",
            "model",
            "execution_time",
            "sparsity",
            "median_library_size",
            "systematic_variation",
            "intra_corr",
            "vendi_score",
            "context_axis",
            "context_values",
        }
        metrics_error = {key: np.nan for key in metrics_error_keys_local}

        error_contexts = (
            [("cell_line", "1")] if dataset_name == "causalDGP" else [("cell_line", "0")]
        )

        final_error_rows = []
        for context_axis, context_values in error_contexts:
            for model in MODELS:
                final_error_rows.append(
                    {
                        **params_dict,  # original sampled params
                        **metrics_error,
                        "dataset": dataset_name,
                        "split_strategy": split_strategy,
                        "diversity_type": diversity_type,
                        "trial_id": trial_id,
                        "model": model,
                        "status": "failed",
                        "error": str(e),
                        "context_axis": context_axis,
                        "context_values": context_values,
                    }
                )
        return final_error_rows


def run_random_sweep(
    dataset_name,
    n_trials,
    output_dir,
    diversity_type="A",
    control_mu=None,
    all_theta=None,
    pert_mu=None,
    gene_names=None,
    rng: np.random.Generator | None = None,
    num_workers=None,
    use_multiprocessing=True,
    split_strategy="in-context",
    trial_start=0,
) -> pd.DataFrame:
    """Run random synthetic sweeps and save results plus error logs."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if trial_start < 0:
        raise ValueError("trial_start must be non-negative.")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if rng is None:
        rng = np.random.default_rng()
    if gene_names is None:
        raise ValueError(
            "gene_names must be provided for synthetic sweeps so perturbations map to genes."
        )
    pid = os.getpid()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{pid}"
    split_strategy_tag = split_strategy.replace("-", "_")
    if dataset_name == "directDGP":
        identifier = f"{dataset_name}_{timestamp}"
    else:
        identifier = f"{dataset_name}_{split_strategy_tag}_{diversity_type}_{timestamp}"
    csv_file = output_dir_path / f"results_{identifier}.csv"
    error_log_file = output_dir_path / f"error_log_{identifier}.txt"

    if use_multiprocessing:
        if num_workers is None:
            num_workers = os.cpu_count()
        print(
            f"Starting in-memory AnnData random parameter sweep with {n_trials} trials "
            f"using {num_workers} worker processes (spawn context)."
        )
    else:
        print(
            f"Starting in-memory AnnData random parameter sweep with {n_trials} trials "
            "using sequential execution."
        )

    tasks_for_pool = []
    # Advance the same parameter stream so separate shards match an unsharded sweep.
    for _ in range(trial_start):
        sample_parameters(PARAM_RANGES, rng)
    for i in range(trial_start, trial_start + n_trials):
        params = sample_parameters(PARAM_RANGES, rng)
        tasks_for_pool.append(
            {
                "trial_id": i,
                "dataset_name": dataset_name,
                "params_dict": params,
                "split_strategy": split_strategy,
                "diversity_type": diversity_type,
                "pid": pid,
            }
        )
    tasks_for_pool = order_tasks_for_pool(tasks_for_pool)

    all_results_data = []

    def checkpoint_results() -> None:
        # Keep completed trials available even if a later trial is interrupted.
        temporary_csv = csv_file.with_suffix(".csv.tmp")
        pd.DataFrame(all_results_data).to_csv(temporary_csv, index=False)
        temporary_csv.replace(csv_file)

    if use_multiprocessing:
        print(
            "Multiprocessing can be memory intensive, so if running into swap, reduce the number of workers."
        )
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(control_mu, all_theta, pert_mu, gene_names),
            maxtasksperchild=100,
        ) as pool:
            print("\nProcessing trials (in-memory AnnData version with worker timing):")
            with tqdm(total=n_trials, desc="Running Trials (in-memory AnnData)") as pbar:
                for result_from_worker in pool.imap_unordered(_pool_worker_timed, tasks_for_pool):
                    all_results_data += result_from_worker
                    checkpoint_results()
                    pbar.update(1)
    else:
        init_worker(control_mu, all_theta, pert_mu, gene_names)
        print("\nProcessing trials (in-memory AnnData version with worker timing):")
        with tqdm(total=n_trials, desc="Running Trials (in-memory AnnData)") as pbar:
            for task in tasks_for_pool:
                result_from_worker = _pool_worker_timed(task)
                all_results_data += result_from_worker
                checkpoint_results()
                pbar.update(1)

    results_df = pd.DataFrame(all_results_data)

    success_count = (
        int(
            results_df[(results_df["status"] == "success") & (results_df["model"] == MODELS[0])][
                "trial_id"
            ].nunique()
        )
        if "status" in results_df
        else 0
    )
    failure_count = n_trials - success_count

    if failure_count > 0 and "status" in results_df:  # Ensure 'status' column exists
        print("")
        failed_trials = results_df[results_df["status"] == "failed"]
        # Define metrics_error keys for excluding them from params logging
        metrics_error_keys = {
            "pearson",
            "pearson_true_degs",
            "pearson_degs",
            "mae",
            "mae_true_degs",
            "mae_degs",
            "mse",
            "mse_true_degs",
            "mse_degs",
            "r2",
            "r2_true_degs",
            "r2_degs",
            "parametric_distance",
            "mmd_distance",
            "fid_distance",
            "des_recall",
            "des_precision",
            "des_jaccard",
            "vendi_score_pred",
            "vendi_score_obs",
            "pds_l1",
            "pds_l2",
            "pds_cosine",
            "sparsity",
            "vendi_score",
            "context_axis",
            "context_values",
        }
        with error_log_file.open("a") as f:
            for _, row in failed_trials.iterrows():
                # Ensure 'trial_id' and 'error' exist in row, provide defaults if not
                trial_id_val = int(row.get("trial_id", -1))
                error_val = row.get("error", "Unknown error")

                error_params = {
                    k: v
                    for k, v in row.items()
                    if k not in metrics_error_keys
                    and k not in ["status", "error", "trial_id", "execution_time"]
                }
                f.write(f"Trial {trial_id_val + 1} failed\n")
                f.write(f"Parameters: {error_params!s}\n")
                f.write(f"Error: {error_val}\n")
                f.write("-" * 80 + "\n")

    if not results_df.empty:
        print(f"\nSweep complete. Results saved to '{csv_file}'")
    else:
        print("\nSweep complete. No results to save.")

    print(f"Success: {success_count}/{n_trials} trials")
    print(f"Failed: {failure_count}/{n_trials} trials")
    if failure_count > 0:
        print(f"See error log for details: {error_log_file}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random sweep simulations.")
    parser.add_argument("--n_trials", type=int, default=4, help="Number of trials to run")
    parser.add_argument(
        "--trial_start",
        type=int,
        default=0,
        help="First global trial ID; shards with the same seed match an unsharded sweep",
    )
    parser.add_argument(
        "--output_dir",
        default=_OUTPUT_DIR,
        help="Directory for completed-trial CSV checkpoints and error logs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes for multiprocessing",
    )
    parser.add_argument("--multiprocessing", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--dataset",
        type=str,
        default="causalDGP",
        choices=["directDGP", "causalDGP"],
        help="Dataset to use for the simulation",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="in-context",
        choices=["in-context", "cross-context"],
        help="Data splitting strategy for evaluation, cross-context is only available for causalDGP",
    )
    parser.add_argument(
        "--diversity_type",
        type=str,
        default="both",
        choices=["A", "b", "both", "none"],
        help="Type of diversity to introduce for causalDGP dataset",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    synthetic_inputs = load_parameter_estimation_inputs()

    print("Using theta estimates from all cells combined")
    print(f"Loaded synthetic parameter estimates from '{ALL_PARAMS_PATH}'.")

    control_mu = synthetic_inputs["control_mu"]
    pert_mu = synthetic_inputs["pert_mu"]
    gene_names = synthetic_inputs["gene_names"]
    all_theta = synthetic_inputs["all_theta"]

    print(f"Using {len(control_mu)} genes for simulation.")

    # Call the final version of run_random_sweep
    print("Running the sweep...")
    results_df = run_random_sweep(
        args.dataset,
        args.n_trials,
        args.output_dir,
        control_mu=control_mu,
        all_theta=all_theta,
        pert_mu=pert_mu,
        gene_names=gene_names,
        rng=rng,
        diversity_type=args.diversity_type,
        num_workers=args.num_workers,  # num_worker should be around 0.6 * RAM / MAX_SPACE_PER_WORK
        use_multiprocessing=args.multiprocessing,
        split_strategy=args.split_strategy,
        trial_start=args.trial_start,
    )
    if results_df.empty or (results_df["status"] != "success").any():
        raise SystemExit(1)
