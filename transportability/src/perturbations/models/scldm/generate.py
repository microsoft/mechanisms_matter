"""Generate held-out perturbations from the trained LDM using a stage-3 checkpoint and classifier-free guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytorch_lightning as pl
import torch

from .build import LDMHParams, VAEHParams, build_ldm_module, build_vae_module
from .data import PerturbseqDataModule
from .runtime import process_generation_output


def _resolve_condition_strategy(ldm_ckpt: str, cli_value: str | None) -> str:
    """
    Authoritative condition_strategy comes from the training artifact, not the CLI.

    ``condition_strategy`` is constructor-only state (it changes DiT.forward_with_cfg and
    _sample_log_size_factors) and is NOT in the checkpoint's state_dict, so a wrong value cannot be
    caught by load_state_dict and would silently corrupt joint size-factor sampling. train_ldm.py
    writes it to a sibling ``hparams.json``; we read that and reject any disagreeing CLI override.
    """
    hp_path = Path(ldm_ckpt).parent / "hparams.json"
    saved = None
    if hp_path.exists():
        saved = json.loads(hp_path.read_text()).get("condition_strategy")
    if saved is None:
        if cli_value is None:
            raise SystemExit(
                f"condition_strategy not found in {hp_path} and no --condition-strategy given; "
                + "cannot safely reconstruct the model."
            )
        print(
            "[generate] WARNING: no hparams.json next to checkpoint; using CLI "
            + f"--condition-strategy={cli_value}"
        )
        return cli_value
    if cli_value is not None and cli_value != saved:
        raise SystemExit(
            f"--condition-strategy={cli_value} disagrees with the trained model "
            + f"(hparams.json says {saved!r}). Omit the flag or pass {saved!r}."
        )
    return saved


def _resolve_class_vocab_sizes(ldm_ckpt: str, dm: PerturbseqDataModule) -> dict[str, int]:
    """
    Reconstruct the DiT condition-embedding sizes used at training time.

    The authoritative source is the sibling ``hparams.json`` written by train_ldm.py, so the
    architecture is rebuilt exactly as trained (required for strict checkpoint loading). Older
    checkpoints without the field fall back to the data-driven sizes from the metadata JSON.
    """
    hp_path = Path(ldm_ckpt).parent / "hparams.json"
    if hp_path.exists():
        saved = json.loads(hp_path.read_text()).get("class_vocab_sizes")
        if saved is not None:
            return {k: int(v) for k, v in saved.items()}
    return dm.class_vocab_sizes


def _resolve_n_genes(ldm_ckpt: str, dm: PerturbseqDataModule) -> int:
    """Reconstruct the VAE gene vocabulary size used at training time."""
    hp_path = Path(ldm_ckpt).parent / "hparams.json"
    if hp_path.exists():
        saved = json.loads(hp_path.read_text()).get("n_genes")
        if saved is not None:
            return int(saved)
    return dm.n_genes


def _load_ldm(
    ldm_ckpt: str,
    condition_strategy: str,
    class_vocab_sizes: dict[str, int],
    n_genes: int,
):
    """
    Reconstruct the LatentDiffusion architecture and load stage-3 weights (VAE + DiT + EMA).

    Uses strict=True: a wrong or incomplete checkpoint (e.g. a VAE checkpoint, or a partial LDM
    checkpoint) MUST fail loudly rather than silently generate from randomly-initialized diffusion /
    EMA weights. The VAE→LDM and LDM→LDM round-trips were verified to load cleanly with strict=True.
    """
    stub_vae = build_vae_module(max_steps=1, warmup_steps=1, n_genes=n_genes).vae_model
    module = build_ldm_module(
        vae_model=stub_vae,
        max_steps=1,
        warmup_steps=1,
        class_vocab_sizes=class_vocab_sizes,
        vae_hp=VAEHParams(),
        hp=LDMHParams(condition_strategy=condition_strategy),
    )
    ckpt = torch.load(ldm_ckpt, map_location="cpu", weights_only=False)
    state = cast(
        "dict[str, torch.Tensor]",
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt,
    )
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise SystemExit(
            f"Refusing to generate: {ldm_ckpt} is not a matching LDM checkpoint "
            + f"(strict load failed). Point --ldm-ckpt at a stage-3 ldm.ckpt.\n{e}"
        ) from e
    module.eval()
    return module


def _resolve(data_dir: Path, path: str) -> Path:
    """Resolve an artifact path: absolute as-is, otherwise relative to --data-dir."""
    p = Path(path)
    return p if p.is_absolute() else data_dir / p


def main() -> None:
    """Generate synthetic data using a pre-trained scLDM model and save the output as an AnnData object."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ldm-ckpt", required=True, help="stage-3 LDM checkpoint (ldm.ckpt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--omega",
        type=float,
        required=True,
        help="CFG guidance weight (Table 3: 1/5/10)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=50)  # generation.yaml default
    ap.add_argument("--test-batch-size", type=int, default=256)
    ap.add_argument(
        "--test-h5ad",
        default="test_asSTATE_hvg.h5ad",
        help="test AnnData filename (relative to --data-dir) or absolute path",
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
    ap.add_argument(
        "--condition-strategy",
        default=None,
        choices=["joint", "mutually_exclusive"],
        help="optional; must match the trained model's hparams.json (else it errors). "
        + "Leave unset to use the saved value.",
    )
    ap.add_argument("--precision", default="32-true")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    d = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    condition_strategy = _resolve_condition_strategy(args.ldm_ckpt, args.condition_strategy)

    dm = PerturbseqDataModule(
        train_h5ad=None,
        test_h5ad=_resolve(d, args.test_h5ad),
        metadata_json=_resolve(d, args.metadata_json),
        mu_size_factor=_resolve(d, args.mu_size_factor),
        sd_size_factor=_resolve(d, args.sd_size_factor),
        test_batch_size=args.test_batch_size,
        seed=args.seed,
    )
    dm.setup("predict")

    class_vocab_sizes = _resolve_class_vocab_sizes(args.ldm_ckpt, dm)
    n_genes = _resolve_n_genes(args.ldm_ckpt, dm)
    module = _load_ldm(args.ldm_ckpt, condition_strategy, class_vocab_sizes, n_genes)
    # classifier-free guidance: per-class weight = omega (joint cell_line+gene). timesteps as paper.
    module.generation_args = {
        "guidance_weight": {"cell_line": args.omega, "gene": args.omega},
        "timesteps": args.timesteps,
    }
    module.inference_args = None

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision=args.precision,
    )
    outputs = trainer.predict(module, datamodule=dm)

    adata = process_generation_output(cast("list[dict[str, torch.Tensor]]", outputs), dm)
    adata.obs["generation_idx"] = args.seed
    save_path = out / f"replogle_generated_{args.seed}.h5ad"
    adata.write(save_path)
    dataset_col = cast("pd.Series", adata.obs["dataset"])
    n_cond = int((dataset_col.astype(str) == "generated_conditional").sum())
    print(f"wrote {save_path}  ({adata.n_obs} cells, {n_cond} generated_conditional)")


if __name__ == "__main__":
    main()
