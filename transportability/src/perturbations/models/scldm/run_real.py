"""
End-to-end scLDM runner for REAL data (VAE stage 2 + LDM stage 3 + generation) in one process.

This is the real-data counterpart of ``smoke_synthetic.py``: same single-process
build -> train -> save -> reload -> generate flow, but it consumes real AnnData objects and
uses the codebase / paper defaults with **no** smoke-test overrides:

- architecture: ``VAEHParams()`` / ``LDMHParams()`` (== ``vae_base.yaml`` / ``ldm_base.yaml``),
- training budget: epoch-driven ``max_steps = num_epochs * (n_cells // batch_size)`` with 10% warmup
  (``compute_training_steps``), the same as ``train_vae.py`` / ``train_ldm.py``,
- generation: ``timesteps=50`` (``generation.yaml`` default), ``omega=1.0`` (paper default).

You pass raw AnnData object(s); this script converts them into the scLDM ``asSTATE`` inputs
(raw counts in ``.X``; ``cell_line`` + ``gene`` condition columns), builds the vocabulary JSON
and the joint log-size-factor pkls, then trains and generates. No reconstruction metric is computed.

Validation: pass a separate ``val_adata`` to mirror the upstream training loop.
Validation runs every ``val_check_every_n_epochs`` epochs (default 10) and the best
checkpoint is selected by ``val_loss`` for both the VAE and the LDM stages.

The transformer attention uses ``flex_attention``, whose backward is GPU-only, so training
requires CUDA.

Use via :func:`run_scldm`, passing raw train/val/test AnnData objects plus the column
mapping and hyperparameters::

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
"""

# anndata / pandas / scipy expose largely-untyped APIs (.X / .obs / .astype / issparse / concat),
# so only the "unknown from untyped library" diagnostics are silenced here; real type errors
# (reportArgumentType / reportCallIssue / ...) stay enabled so genuine bugs are still caught.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.sparse import issparse

from .build import (
    LDMHParams,
    VAEHParams,
    build_ldm_module,
    build_vae_module,
    compute_training_steps,
)
from .data import PerturbseqDataModule
from .generate import (
    _load_ldm,  # pyright: ignore[reportPrivateUsage]
    _resolve_class_vocab_sizes,
)
from .runtime import process_generation_output

CONTROL_TOKEN = "nontargeting"


def _sanitize(value: object) -> str:
    """Strip characters that would break the joint ``cell_line_gene`` token split on ``_``."""
    return re.sub(r"[^0-9a-zA-Z]", "", str(value))


def _counts_and_conditions(
    adata: ad.AnnData,
    *,
    perturbation_column: str,
    context_key: str,
    control_label: str,
    counts_layer: str | None,
) -> ad.AnnData:
    """Return a fresh AnnData with float32 raw counts in ``.X`` and ``cell_line``/``gene`` obs cols."""
    if counts_layer is None:
        counts_raw: Any = adata.X
    else:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"Requested counts layer {counts_layer!r} is missing from the input AnnData."
            )
        counts_raw = adata.layers[counts_layer]
    counts = (
        counts_raw.astype(np.float32, copy=True)
        if issparse(counts_raw)
        else np.asarray(counts_raw, dtype=np.float32).copy()
    )
    out = ad.AnnData(
        X=counts,
        obs=cast("pd.DataFrame", adata.obs).copy(),
        var=cast("pd.DataFrame", adata.var).copy(),
    )

    out_obs = cast("pd.DataFrame", out.obs)
    context = out_obs[context_key].astype(str).map(_sanitize)
    gene = (
        out_obs[perturbation_column]
        .astype(str)
        .map(lambda p: CONTROL_TOKEN if p == control_label else _sanitize(p))
    )
    out.obs["cell_line"] = context.astype("category")
    out.obs["gene"] = gene.astype("category")
    return out


def _library_sizes(adata: ad.AnnData) -> np.ndarray:
    """Return per-cell library sizes without densifying sparse count matrices."""
    matrix: Any = adata.X
    if issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel()
    return np.asarray(matrix).sum(axis=1)


def _grouped_mean_std(
    values: np.ndarray, groups: pd.Series
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute group statistics with a valid positive scale for Normal sampling."""
    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    minimum_scale = float(np.finfo(np.float32).eps)
    for group in groups.unique():
        group_values = values[(groups == group).to_numpy()]
        group_key = str(group)
        means[group_key] = float(group_values.mean())
        standard_deviations[group_key] = max(float(group_values.std()), minimum_scale)
    return means, standard_deviations


def write_asstate_inputs(
    train_adata: ad.AnnData,
    test_adata: ad.AnnData,
    out_dir: Path,
    *,
    val_adata: ad.AnnData,
    perturbation_column: str,
    context_key: str,
    control_label: str,
    counts_layer: str | None,
) -> dict[str, Path]:
    """
    Convert real AnnData object(s) into the scLDM ``asSTATE`` inputs and write them to disk.

    Produces train/validation/test AnnData objects (raw counts in ``.X``; ``cell_line`` +
    ``gene`` condition columns), the vocabulary JSON, and the joint log-size-factor pkls. The
    label vocabulary is taken over the union of train + test + val so held-out labels present only
    in a held-out split are still known to the model, but the log-size-factor statistics are
    computed from the fit split (train + val) ONLY, so no test-set library sizes leak into
    generation.
    """
    reference_var_names = np.asarray(train_adata.var_names).astype(str)
    for split_name, split_adata in (("val", val_adata), ("test", test_adata)):
        if not np.array_equal(reference_var_names, np.asarray(split_adata.var_names).astype(str)):
            raise ValueError(
                f"scLDM requires train, val, and test genes in the same order; {split_name} differs."
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    train = _counts_and_conditions(
        train_adata,
        perturbation_column=perturbation_column,
        context_key=context_key,
        control_label=control_label,
        counts_layer=counts_layer,
    )
    test = _counts_and_conditions(
        test_adata,
        perturbation_column=perturbation_column,
        context_key=context_key,
        control_label=control_label,
        counts_layer=counts_layer,
    )

    val = _counts_and_conditions(
        val_adata,
        perturbation_column=perturbation_column,
        context_key=context_key,
        control_label=control_label,
        counts_layer=counts_layer,
    )

    paths = {
        "train_h5ad": out_dir / "train_asSTATE_hvg.h5ad",
        "test_h5ad": out_dir / "test_asSTATE_hvg.h5ad",
        "metadata_json": out_dir / "metadata_train.json",
        "mu_size_factor": out_dir / "log_size_factor_mu.pkl",
        "sd_size_factor": out_dir / "log_size_factor_sd.pkl",
        "val_h5ad": out_dir / "val_asSTATE_hvg.h5ad",
    }
    train.write_h5ad(paths["train_h5ad"])
    test.write_h5ad(paths["test_h5ad"])
    val.write_h5ad(paths["val_h5ad"])

    # Test labels define requested conditions, but no test expression enters fitted statistics.
    full_obs = pd.concat(
        [
            train.obs[["cell_line", "gene"]],
            val.obs[["cell_line", "gene"]],
            test.obs[["cell_line", "gene"]],
        ],
        ignore_index=True,
    )
    labels = {
        "cell_line": sorted(full_obs["cell_line"].astype(str).unique().tolist()),
        "gene": sorted(full_obs["gene"].astype(str).unique().tolist()),
    }
    paths["metadata_json"].write_text(
        json.dumps({"genes": list(map(str, train.var_names)), "labels": labels}, indent=2),
        encoding="utf-8",
    )

    # Library-size statistics are fitted strictly on train + val.
    fit_obs = pd.concat(
        [train.obs[["cell_line", "gene"]], val.obs[["cell_line", "gene"]]],
        ignore_index=True,
    )
    log_library_sizes = np.log(np.concatenate([_library_sizes(train), _library_sizes(val)]) + 1e-8)
    joint_conditions = fit_obs["cell_line"].astype(str) + "_" + fit_obs["gene"].astype(str)
    mu, sd = _grouped_mean_std(log_library_sizes, joint_conditions)
    cell_lines = fit_obs["cell_line"].astype(str)
    mu_cell_line, sd_cell_line = _grouped_mean_std(log_library_sizes, cell_lines)
    joint_key = "cell_line_gene"
    with paths["mu_size_factor"].open("wb") as f:
        pickle.dump({joint_key: mu, "cell_line": mu_cell_line}, f)
    with paths["sd_size_factor"].open("wb") as f:
        pickle.dump({joint_key: sd, "cell_line": sd_cell_line}, f)

    return paths


def _train(
    module: pl.LightningModule,
    dm: pl.LightningDataModule,
    *,
    max_steps: int,
    ckpt_dir: Path,
    precision: str,
    val_check_every_n_epochs: int,
    resume: bool = True,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """
    Train ``module`` mirroring the upstream scLDM training loop.

    Validation runs every ``val_check_every_n_epochs`` epochs and the checkpoint is selected by
    ``val_loss`` (min); a rolling ``last.ckpt`` is also kept for preemption-resilient resume. When
    ``resume`` is False, an existing ``last.ckpt`` is ignored so training always starts fresh.
    """
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch}",
    )
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=True,
        callbacks=[ckpt_cb],
        precision=cast("Any", precision),
        check_val_every_n_epoch=max(1, val_check_every_n_epochs),
        num_sanity_val_steps=0,
    )
    resume_ckpt = ckpt_dir / "last.ckpt"
    trainer.fit(
        module,
        datamodule=dm,
        ckpt_path=str(resume_ckpt) if resume and resume_ckpt.exists() else None,
    )
    return trainer, ckpt_cb


def _load_best_into(module: pl.LightningModule, ckpt_cb: ModelCheckpoint) -> None:
    """Load the best-by-val_loss checkpoint weights back into ``module`` (upstream selection)."""
    best = getattr(ckpt_cb, "best_model_path", "")
    if best and Path(best).exists():
        state = torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
        module.load_state_dict(state)
        print(f"[scldm-run] loaded best checkpoint by val_loss -> {best}")
    else:
        print("[scldm-run] no best checkpoint found; keeping last weights")


def _normalize_log1p(counts: np.ndarray, target_sum: float) -> np.ndarray:
    """Match the benchmark eval layer: per-cell normalize_total(target_sum) then log1p."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(counts / total * target_sum).astype(np.float32)


def _build_test_aligned_preds(
    test_adata: ad.AnnData,
    gen_conditional: ad.AnnData,
    *,
    perturbation_column: str,
    context_key: str,
    control_label: str,
    counts_layer: str | None,
    normalized_target_sum: float | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return generated predictions aligned exactly to the test-cell order."""
    if counts_layer is None:
        raw_src: Any = test_adata.X
    else:
        if counts_layer not in test_adata.layers:
            raise KeyError(f"Requested counts layer {counts_layer!r} is missing from test_adata.")
        raw_src = test_adata.layers[counts_layer]

    raw = np.asarray(raw_src.toarray() if issparse(raw_src) else raw_src, dtype=np.float32)
    truths = (
        raw.copy()
        if normalized_target_sum is None
        else _normalize_log1p(raw, normalized_target_sum)
    )

    generated_matrix: Any = gen_conditional.X
    generated_counts = np.asarray(
        generated_matrix.toarray() if issparse(generated_matrix) else generated_matrix,
        dtype=np.float32,
    )
    if generated_counts.shape != truths.shape:
        raise ValueError(
            f"Generated shape {generated_counts.shape} does not match test shape {truths.shape}."
        )
    if not np.array_equal(
        np.asarray(gen_conditional.var_names).astype(str),
        np.asarray(test_adata.var_names).astype(str),
    ):
        raise ValueError("Generated genes do not match test genes in name and order.")
    generated_values = (
        generated_counts
        if normalized_target_sum is None
        else _normalize_log1p(generated_counts, normalized_target_sum)
    )

    test_obs = cast("pd.DataFrame", test_adata.obs)
    test_contexts = test_obs[context_key].astype(str).map(_sanitize).to_numpy()
    test_perturbations = (
        test_obs[perturbation_column]
        .astype(str)
        .map(lambda value: CONTROL_TOKEN if value == control_label else _sanitize(value))
        .to_numpy()
    )
    generated_obs = cast("pd.DataFrame", gen_conditional.obs)
    generated_contexts = generated_obs["cell_line"].astype(str).to_numpy()
    generated_perturbations = generated_obs["gene"].astype(str).to_numpy()
    if not np.array_equal(test_contexts, generated_contexts) or not np.array_equal(
        test_perturbations, generated_perturbations
    ):
        raise ValueError("Generated conditions are not aligned one-to-one with test cells.")

    predictions = truths.copy()
    perturbed = test_perturbations != CONTROL_TOKEN
    predictions[perturbed] = generated_values[perturbed]
    perturbation_names = test_obs[perturbation_column].astype(str).tolist()
    return predictions, truths, perturbation_names


def run_scldm(
    train_adata: ad.AnnData,
    val_adata: ad.AnnData,
    test_adata: ad.AnnData,
    out_dir: str | Path = ".",
    perturbation_column: str = "perturbation",
    context_key: str = "cell_type",
    control_label: str = "control",
    num_epochs: int = 100,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    precision: str = "32-true",  # PyTorch Lightning trainer precision (e.g. "bf16-mixed") for faster
    condition_strategy: str = "joint",
    max_steps: int | None = None,
    val_check_every_n_epochs: int = 10,
    resume: bool = True,
    timesteps: int = 50,
    omega: float = 1.0,
    counts_layer: str | None = "counts",
    normalized_target_sum: float | None = 1e4,
) -> dict[str, Any]:
    """
    Run the full scLDM pipeline on in-memory AnnData: convert -> train VAE -> train LDM -> generate.

    Mirrors ``state_gene.run_state_gene``: pass raw train/val/test AnnData objects plus the column
    mapping and training/generation hyperparameters (instead of CLI args / file paths).

    Args:
        train_adata: raw train AnnData (counts in ``.X`` or a ``counts`` layer).
        val_adata: raw validation AnnData.
        test_adata: raw test AnnData.
        out_dir: directory for asSTATE inputs, checkpoints, and the generated h5ad.
        perturbation_column: obs column mapped to the ``gene`` condition.
        context_key: obs column mapped to the ``cell_line`` condition.
        control_label: value in ``perturbation_column`` that means control.
        num_epochs: number of training epochs (drives the max-steps budget).
        batch_size: training batch size.
        num_workers: DataLoader worker processes.
        seed: RNG seed for reproducibility.
        precision: PyTorch Lightning trainer precision (e.g. ``"bf16-mixed"``).
        condition_strategy: DiT conditioning ("joint" for Replogle, else "mutually_exclusive").
        max_steps: override the epoch-derived training-step budget.
        val_check_every_n_epochs: validation cadence (best checkpoint by ``val_loss``).
        resume: if True (default), resume each stage from an existing ``last.ckpt`` in ``out_dir``
            (preemption-resilient); set False to always train fresh (reproducible re-runs).
        timesteps: ODE sampling steps for generation.
        omega: classifier-free-guidance weight for generation.
        counts_layer: raw-counts layer read from the input AnnData objects. Set to None to use X.
        normalized_target_sum: per-cell total used for normalize-total plus log1p predictions.
            Set to None to return predictions in raw-count space.

    Returns:
        Dict with:
            - ``preds``: ``(n_test_cells, n_genes)`` prediction matrix row-aligned to ``test_adata``
              in the eval-layer space; perturbed rows are generated conditional cells matched per
              condition (no averaging), control rows are ground truth (mirrors ``run_state_gene``).
            - ``truths``: ``(n_test_cells, n_genes)`` eval-layer expression of ``test_adata``.
            - ``pert_names``: per-test-cell perturbation labels (``list[str]``).
            - ``paths``: mapping of asSTATE input names to written file paths (train/test/val
              h5ad, metadata JSON, and the mu/sd log-size-factor pkls).
            - ``vae_ckpt``: path to the saved VAE (stage 2) checkpoint (``<out_dir>/vae.ckpt``).
            - ``ldm_ckpt``: path to the saved LDM (stage 3) checkpoint (``<out_dir>/ldm.ckpt``).
            - ``generated_adata``: the generated :class:`~anndata.AnnData` (conditional +
              unconditional cells over the test-set conditions).
            - ``generated_path``: path to the written generated h5ad (``generated_<seed>.h5ad``).
            - ``n_conditional``: number of generated conditional cells.
            - ``n_unconditional``: number of generated unconditional cells.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "scLDM training requires CUDA because flex_attention backward is GPU-only."
        )
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: convert raw AnnData(s) -> asSTATE inputs --------------------------------------
    asstate = out / "asstate"
    paths = write_asstate_inputs(
        train_adata,
        test_adata,
        asstate,
        val_adata=val_adata,
        perturbation_column=perturbation_column,
        context_key=context_key,
        control_label=control_label,
        counts_layer=counts_layer,
    )
    print(f"[scldm-run] wrote asSTATE inputs -> {asstate}")

    dm = PerturbseqDataModule(
        train_h5ad=paths["train_h5ad"],
        test_h5ad=paths["test_h5ad"],
        metadata_json=paths["metadata_json"],
        mu_size_factor=paths["mu_size_factor"],
        sd_size_factor=paths["sd_size_factor"],
        val_h5ad=paths["val_h5ad"],
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    dm.setup("fit")

    print(
        "[scldm-run] validation: separate val set every "
        + f"{val_check_every_n_epochs} epochs, best checkpoint by val_loss"
    )

    max_steps_eff, warmup = compute_training_steps(dm.n_cells, batch_size, num_epochs)
    if max_steps is not None:
        max_steps_eff, warmup = max_steps, max(1, int(0.1 * max_steps))
    print(
        f"[scldm-run] n_cells={dm.n_cells} batch_size={batch_size} "
        + f"num_epochs={num_epochs} -> max_steps={max_steps_eff} warmup={warmup}"
    )

    # ---- Stage 2: VAE -------------------------------------------------------------------------
    vae_hp = VAEHParams()
    vae_module = build_vae_module(
        max_steps=max_steps_eff,
        warmup_steps=warmup,
        n_genes=dm.n_genes,
        hp=vae_hp,
    )
    vae_trainer, vae_ckpt_cb = _train(
        vae_module,
        dm,
        max_steps=max_steps_eff,
        ckpt_dir=out / "vae_ckpts",
        precision=precision,
        val_check_every_n_epochs=val_check_every_n_epochs,
        resume=resume,
    )
    _load_best_into(vae_module, vae_ckpt_cb)
    vae_trainer.save_checkpoint(out / "vae.ckpt")
    print(f"[scldm-run] saved VAE checkpoint -> {out / 'vae.ckpt'}")

    # ---- Stage 3: LDM (frozen VAE tokenizer + DiT) --------------------------------------------
    vae_model = vae_module.vae_model
    for p in vae_model.parameters():
        p.requires_grad = False
    vae_model.eval()

    class_vocab_sizes = dm.class_vocab_sizes
    ldm_hp = LDMHParams(condition_strategy=condition_strategy)
    ldm_module = build_ldm_module(
        vae_model=vae_model,
        max_steps=max_steps_eff,
        warmup_steps=warmup,
        class_vocab_sizes=class_vocab_sizes,
        vae_hp=vae_hp,
        hp=ldm_hp,
    )
    ldm_trainer, ldm_ckpt_cb = _train(
        ldm_module,
        dm,
        max_steps=max_steps_eff,
        ckpt_dir=out / "ldm_ckpts",
        precision=precision,
        val_check_every_n_epochs=val_check_every_n_epochs,
        resume=resume,
    )
    _load_best_into(ldm_module, ldm_ckpt_cb)
    ldm_ckpt = out / "ldm.ckpt"
    ldm_trainer.save_checkpoint(ldm_ckpt)
    (out / "hparams.json").write_text(
        json.dumps(
            {
                "stage": "ldm",
                "n_cells": dm.n_cells,
                "n_genes": dm.n_genes,
                "max_steps": max_steps_eff,
                "warmup": warmup,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "condition_strategy": condition_strategy,
                "class_vocab_sizes": class_vocab_sizes,
                "ldm_hparams": vars(ldm_hp),
            },
            indent=2,
            default=str,
        )
    )
    print(f"[scldm-run] saved LDM checkpoint -> {ldm_ckpt}")

    # ---- Generation: reload through the real generate.py helpers (strict load) -----------------
    resolved_sizes = _resolve_class_vocab_sizes(str(ldm_ckpt), dm)
    gen_module = _load_ldm(str(ldm_ckpt), condition_strategy, resolved_sizes, dm.n_genes)
    gen_module.generation_args = {
        "guidance_weight": {"cell_line": omega, "gene": omega},
        "timesteps": timesteps,
    }
    gen_module.inference_args = None

    gen_trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision="32-true",
    )
    outputs = gen_trainer.predict(gen_module, datamodule=dm)
    adata_gen = process_generation_output(cast("list[dict[str, torch.Tensor]]", outputs), dm)
    adata_gen.obs["generation_idx"] = seed
    save_path = out / f"generated_{seed}.h5ad"
    adata_gen.write(save_path)
    n_cond = int((adata_gen.obs["dataset"].astype(str) == "generated_conditional").sum())
    n_uncond = int((adata_gen.obs["dataset"].astype(str) == "generated_unconditional").sum())
    print(
        f"[scldm-run] wrote {save_path}  ({adata_gen.n_obs} cells, "
        + f"{n_cond} conditional, {n_uncond} unconditional; omega={omega} timesteps={timesteps})"
    )

    # Test-aligned predictions (mirrors run_state_gene): per-condition, per-cell, no averaging.
    gen_conditional = adata_gen[adata_gen.obs["dataset"].astype(str) == "generated_conditional"]
    preds, truths, pert_names = _build_test_aligned_preds(
        test_adata,
        gen_conditional,
        perturbation_column=perturbation_column,
        context_key=context_key,
        control_label=control_label,
        counts_layer=counts_layer,
        normalized_target_sum=normalized_target_sum,
    )

    return {
        "preds": preds,
        "truths": truths,
        "pert_names": pert_names,
        "paths": paths,
        "vae_ckpt": out / "vae.ckpt",
        "ldm_ckpt": ldm_ckpt,
        "generated_adata": adata_gen,
        "generated_path": save_path,
        "n_conditional": n_cond,
        "n_unconditional": n_uncond,
    }
