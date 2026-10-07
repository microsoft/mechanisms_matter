"""Stage 3 (Hydra-free) training script for the scLDM latent flow-matching DiT on Replogle."""

# PyTorch Lightning / torch checkpoint APIs are largely untyped (torch.load -> Any,
# ModelCheckpoint.best_model_path/.best_model_score), so only the "unknown from untyped library"
# diagnostics are silenced here; real type errors (reportArgumentType / reportCallIssue) stay on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .build import (
    LDMHParams,
    VAEHParams,
    build_ldm_module,
    build_vae_module,
    compute_training_steps,
)
from .data import PerturbseqDataModule


def _resolve(data_dir: Path, path: str) -> Path:
    """Resolve an artifact path: absolute as-is, otherwise relative to --data-dir."""
    p = Path(path)
    return p if p.is_absolute() else data_dir / p


def _load_frozen_vae(vae_ckpt: str, n_genes: int):
    """
    Reconstruct the TransformerVAE and load stage-2 weights (VAE Lightning checkpoint).

    strict=True so a wrong/incomplete checkpoint fails loudly instead of leaving the tokenizer
    partly random.
    """
    stub = build_vae_module(max_steps=1, warmup_steps=1, n_genes=n_genes)  # architecture only
    ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        stub.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise SystemExit(
            f"Refusing to train LDM: {vae_ckpt} is not a matching VAE checkpoint "
            + f"(strict load failed). Point --vae-ckpt at a stage-2 vae.ckpt.\n{e}"
        ) from e
    vae_model = stub.vae_model
    for p in vae_model.parameters():
        p.requires_grad = False
    vae_model.eval()
    return vae_model


def main() -> None:
    """Train the scLDM latent flow-matching DiT on Replogle using a frozen stage-2 VAE."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vae-ckpt", required=True, help="stage-2 VAE checkpoint (vae.ckpt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--train-h5ad",
        default="train_asSTATE_hvg.h5ad",
        help="train AnnData filename (relative to --data-dir) or absolute path",
    )
    ap.add_argument(
        "--test-h5ad",
        default="test_asSTATE_hvg.h5ad",
        help="test AnnData filename (relative to --data-dir) or absolute path",
    )
    ap.add_argument(
        "--val-h5ad",
        required=True,
        help=(
            "validation AnnData filename (relative to --data-dir) or absolute path used for "
            "best-val checkpoint selection"
        ),
    )
    ap.add_argument(
        "--metadata-json",
        default="replogle_train.json",
        help="gene/label vocabulary JSON (relative to --data-dir) or absolute path",
    )
    ap.add_argument(
        "--mu-size-factor",
        default="replogle_log_size_factor_munew.pkl",
        help="log size-factor mean pkl (relative to --data-dir) or absolute path",
    )
    ap.add_argument(
        "--sd-size-factor",
        default="replogle_log_size_factor_sdnew.pkl",
        help="log size-factor std pkl (relative to --data-dir) or absolute path",
    )
    ap.add_argument("--num-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)  # ldm_base.yaml batch_size
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument(
        "--condition-strategy", default="joint", choices=["joint", "mutually_exclusive"]
    )
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--ckpt-every-n-epochs",
        type=int,
        default=1,
        help="write a rolling last.ckpt every N epochs for preemption-resilient resume",
    )
    ap.add_argument(
        "--val-check-every-n-epochs",
        type=int,
        default=10,
        help="run validation every N epochs (upstream default: 10)",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    d = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dm = PerturbseqDataModule(
        train_h5ad=_resolve(d, args.train_h5ad),
        test_h5ad=_resolve(d, args.test_h5ad),
        metadata_json=_resolve(d, args.metadata_json),
        mu_size_factor=_resolve(d, args.mu_size_factor),
        sd_size_factor=_resolve(d, args.sd_size_factor),
        val_h5ad=_resolve(d, args.val_h5ad),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dm.setup("fit")

    max_steps, warmup = compute_training_steps(dm.n_cells, args.batch_size, args.num_epochs)
    if args.max_steps is not None:
        max_steps, warmup = args.max_steps, max(1, int(0.1 * args.max_steps))

    vae_model = _load_frozen_vae(args.vae_ckpt, dm.n_genes)
    ldm_hp = LDMHParams(condition_strategy=args.condition_strategy)
    module = build_ldm_module(
        vae_model=vae_model,
        max_steps=max_steps,
        warmup_steps=warmup,
        class_vocab_sizes=dm.class_vocab_sizes,
        vae_hp=VAEHParams(),
        hp=ldm_hp,
    )

    # Best-val selection + preemption-resilient resume in one callback: keep the single lowest
    # ``val_loss`` checkpoint (validation runs on the dedicated --val-h5ad split) and a rolling
    # last.ckpt. On low-priority A100 a multi-day fit WILL be preempted; resuming from last.ckpt
    # caps a preemption so the AML retry does not restart from step 0.
    ckpt_dir = out / "ckpts"
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch}",
    )
    trainer_kwargs: dict[str, Any] = dict(
        max_steps=max_steps,
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=True,
        callbacks=[ckpt_cb],
        precision=args.precision,
        check_val_every_n_epoch=max(1, args.val_check_every_n_epochs),
        num_sanity_val_steps=0,
    )
    if args.smoke:
        trainer_kwargs.update(
            max_steps=2,
            limit_train_batches=2,
            limit_val_batches=2,
            check_val_every_n_epoch=1,
            precision="32-true",
        )
    trainer = pl.Trainer(**trainer_kwargs)

    resume = ckpt_dir / "last.ckpt"
    trainer.fit(module, datamodule=dm, ckpt_path=str(resume) if resume.exists() else None)

    # Persist the best-val model as ldm.ckpt (fall back to final weights if validation never ran).
    ckpt = out / "ldm.ckpt"
    if ckpt_cb.best_model_path:
        state = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        module.load_state_dict(state)
        best_score = ckpt_cb.best_model_score
        score_str = f"{float(best_score):.4f}" if best_score is not None else "n/a"
        print(
            f"loaded best-val LDM (val_loss={score_str}) "
            + f"<- {Path(ckpt_cb.best_model_path).name}"
        )
    else:
        print("no best-val checkpoint recorded; saving final training weights")
    trainer.save_checkpoint(ckpt)
    (out / "hparams.json").write_text(
        json.dumps(
            {
                "stage": "ldm",
                "n_cells": dm.n_cells,
                "n_genes": dm.n_genes,
                "max_steps": max_steps,
                "warmup": warmup,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "condition_strategy": args.condition_strategy,
                "class_vocab_sizes": dm.class_vocab_sizes,
                "ldm_hparams": vars(ldm_hp),
            },
            indent=2,
            default=str,
        )
    )
    print(f"saved LDM checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
