"""train the scLDM VAE on Replogle."""

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

from .build import VAEHParams, build_vae_module, compute_training_steps
from .data import PerturbseqDataModule


def _resolve(data_dir: Path, path: str) -> Path:
    """Resolve an artifact path: absolute as-is, otherwise relative to --data-dir."""
    p = Path(path)
    return p if p.is_absolute() else data_dir / p


def main() -> None:
    """Train the scLDM VAE on Replogle using argparse and PyTorch Lightning."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        required=True,
        help="dir with train/test_asSTATE_hvg.h5ad + replogle_train.json + size-factor pkls",
    )
    ap.add_argument("--out-dir", required=True, help="output dir for vae.ckpt")
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
    ap.add_argument("--batch-size", type=int, default=128)  # vae_base.yaml batch_size
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override computed max_steps (else num_epochs * steps/epoch)",
    )
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
    ap.add_argument("--smoke", action="store_true", help="fast_dev_run-style tiny loop")
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

    hp = VAEHParams()
    module = build_vae_module(max_steps=max_steps, warmup_steps=warmup, n_genes=dm.n_genes, hp=hp)

    # Best-val selection + preemption-resilient resume in one callback: keep the single lowest
    # ``val_loss`` checkpoint (validation runs on the dedicated --val-h5ad split) and a rolling
    # last.ckpt for resume.
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

    # Persist the best-val model as vae.ckpt (fall back to final weights if validation never ran).
    ckpt = out / "vae.ckpt"
    if ckpt_cb.best_model_path:
        state = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        module.load_state_dict(state)
        best_score = ckpt_cb.best_model_score
        score_str = f"{float(best_score):.4f}" if best_score is not None else "n/a"
        print(
            f"loaded best-val VAE (val_loss={score_str}) "
            + f"<- {Path(ckpt_cb.best_model_path).name}"
        )
    else:
        print("no best-val checkpoint recorded; saving final training weights")
    trainer.save_checkpoint(ckpt)
    (out / "hparams.json").write_text(
        json.dumps(
            {
                "stage": "vae",
                "n_cells": dm.n_cells,
                "n_genes": dm.n_genes,
                "max_steps": max_steps,
                "warmup": warmup,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "vae_hparams": vars(hp),
            },
            indent=2,
            default=str,
        )
    )
    print(f"saved VAE checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
