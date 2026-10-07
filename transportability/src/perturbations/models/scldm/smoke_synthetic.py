"""
Synthetic-data smoke test for the scLDM pipeline (VAE stage 2 + LDM stage 3 + generation).

Generates a small synthetic perturbation dataset (reusing
``models.test_synthentic_data.generate_synthetic_perturbation_data``), splits it into
train/val/test AnnData objects, and runs the whole pipeline through :func:`run_real.run_scldm`
(convert -> train VAE -> train LDM -> generate). It then scores generation with two metrics: a
coarse per-gene mean-expression Pearson correlation of the generated vs. real perturbed cells
(baseline-dominated, so conditional and unconditional score similarly), and a perturbation-effect
correlation of the generated vs. real mean shift from control (sensitive to whether conditioning
captures perturbation-specific effects).

The transformer attention uses ``flex_attention``, whose backward is GPU-only, so ``run_scldm``
trains on CUDA; this smoke test therefore requires a GPU. It is a fast end-to-end sanity check,
not a benchmark.

Run with::

    python -m models.scldm.smoke_synthetic --out-dir /tmp/scldm_smoke
"""

# anndata / pandas / scipy expose largely-untyped APIs (.X / .obs / .astype / issparse), so only
# the "unknown from untyped library" diagnostics are silenced here; real type errors stay enabled.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false
from __future__ import annotations

import argparse
import re
from typing import Any

import anndata as ad
import numpy as np
import torch
from scipy.sparse import issparse

from ..test_synthentic_data import generate_synthetic_perturbation_data
from .run_real import run_scldm

CONTROL_TOKEN = "nontargeting"


def _counts(adata: ad.AnnData) -> np.ndarray:
    """Return dense raw counts from ``.layers['counts']`` if present, else ``.X``."""
    x: Any = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return np.asarray(x.toarray() if issparse(x) else x)


def _sanitize(value: object) -> str:
    """Match ``run_scldm``'s label sanitization so real perturbation names align with generated ones."""
    return re.sub(r"[^0-9a-zA-Z]", "", str(value))


def _split(
    adata: ad.AnnData, *, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 0
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Random 3-way split into (train, val, test) AnnData objects fed to ``run_scldm``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(adata.n_obs)
    n_test = max(1, round(test_frac * adata.n_obs))
    n_val = max(1, round(val_frac * adata.n_obs))
    test = adata[perm[:n_test]].copy()
    val = adata[perm[n_test : n_test + n_val]].copy()
    train = adata[perm[n_test + n_val :]].copy()
    return train, val, test


def _perturbation_effect_correlation(
    test_counts: np.ndarray,
    test_pert: np.ndarray,
    gen_counts: np.ndarray,
    gen_gene: np.ndarray,
    gen_dataset: np.ndarray,
    *,
    control_label: str,
) -> float:
    """
    Mean per-perturbation correlation of the generated vs. real perturbation effect.

    For each perturbation the effect is the per-gene mean-count shift from control
    (``mean(perturbed) - mean(control)``); correlating the real and generated effects across genes
    tests whether conditioning reproduces perturbation-specific shifts, unlike the baseline-dominated
    bulk metric. Real labels are sanitized to match the canonical generated ``gene`` tokens. Returns
    ``nan`` if no perturbation could be matched.
    """
    cond = gen_dataset == "generated_conditional"
    real_ctrl = test_pert == control_label
    gen_ctrl = cond & (gen_gene == CONTROL_TOKEN)
    if real_ctrl.sum() == 0 or gen_ctrl.sum() == 0:
        return float("nan")
    real_ctrl_mean = test_counts[real_ctrl].mean(axis=0)
    gen_ctrl_mean = gen_counts[gen_ctrl].mean(axis=0)

    rs: list[float] = []
    for pert in np.unique(test_pert[test_pert != control_label]):
        real_mask = test_pert == pert
        gen_mask = cond & (gen_gene == _sanitize(pert))
        if real_mask.sum() == 0 or gen_mask.sum() == 0:
            continue
        real_delta = test_counts[real_mask].mean(axis=0) - real_ctrl_mean
        gen_delta = gen_counts[gen_mask].mean(axis=0) - gen_ctrl_mean
        if real_delta.std() == 0 or gen_delta.std() == 0:
            continue
        rs.append(float(np.corrcoef(real_delta, gen_delta)[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


def _report_generation_quality(
    result: dict[str, Any],
    test_adata: ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str,
) -> None:
    """
    Score conditional generation on the returned ``generated_adata``.

    Reports two metrics: (1) a coarse per-gene mean-expression Pearson correlation of the generated
    perturbed cells vs. the real perturbed test cells (conditional vs. the unconditional CFG-null
    samples) — baseline-dominated, so conditional ≈ unconditional; and (2) a perturbation-effect
    correlation of the generated vs. real mean shift from control, which is sensitive to whether
    conditioning captures perturbation-specific effects.
    """
    gen = result["generated_adata"]
    gen_counts = _counts(gen)
    dataset = gen.obs["dataset"].astype(str).to_numpy()
    gen_gene = gen.obs["gene"].astype(str).to_numpy()
    gen_perturbed = gen_gene != CONTROL_TOKEN
    if gen_perturbed.sum() == 0:
        gen_perturbed = np.ones(gen.n_obs, dtype=bool)

    test_counts = _counts(test_adata)
    test_pert = test_adata.obs[perturbation_column].astype(str).to_numpy()
    test_perturbed = test_pert != control_label
    if test_perturbed.sum() == 0:
        test_perturbed = np.ones(test_adata.n_obs, dtype=bool)
    true_mean = test_counts[test_perturbed].mean(axis=0)

    def _mean_for(tag: str) -> np.ndarray:
        mask = (dataset == tag) & gen_perturbed
        return gen_counts[mask].mean(axis=0)

    r_cond = float(np.corrcoef(true_mean, _mean_for("generated_conditional"))[0, 1])
    r_uncond = float(np.corrcoef(true_mean, _mean_for("generated_unconditional"))[0, 1])
    print(
        f"[scldm-smoke] LDM conditional per-gene mean-expression Pearson r = {r_cond:.3f} "
        + f"(unconditional r = {r_uncond:.3f}, perturbed_test={int(test_perturbed.sum())})"
    )

    r_delta = _perturbation_effect_correlation(
        test_counts,
        test_pert,
        gen_counts,
        gen_gene,
        dataset,
        control_label=control_label,
    )
    delta_str = "n/a (no matched perturbations)" if np.isnan(r_delta) else f"{r_delta:.3f}"
    print(f"[scldm-smoke] perturbation-effect (Δ vs control) mean per-gene Pearson r = {delta_str}")


def main() -> None:
    """Run the scLDM synthetic smoke test end to end via ``run_scldm``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="dir for asSTATE inputs + ckpts + outputs")
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument(
        "--n-genes",
        type=int,
        default=200,
        help="must be >= 200; the synthetic generator uses fixed gene indices up to ~140",
    )
    ap.add_argument("--n-perturbations", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=100, help="training-step budget per stage")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--precision",
        default="32-true",
        help="trainer precision; default 32-true (float32) for reproducible smoke scores",
    )
    ap.add_argument("--timesteps", type=int, default=4, help="ODE sampling steps for generation")
    ap.add_argument("--omega", type=float, default=1.0, help="CFG guidance weight for generation")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume each stage from an existing last.ckpt in --out-dir (default: train fresh)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "[scldm-smoke] run_scldm requires CUDA (flex_attention backward is GPU-only)"
        )

    adata = generate_synthetic_perturbation_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_perturbations=args.n_perturbations,
        context_key="cell_type",
        control_label="control",
        perturbation_column="perturbation",
        seed=args.seed,
    )
    train, val, test = _split(adata, seed=args.seed)
    print(f"[scldm-smoke] split -> train={train.n_obs} val={val.n_obs} test={test.n_obs}")

    result = run_scldm(
        train_adata=train,
        val_adata=val,
        test_adata=test,
        out_dir=args.out_dir,
        perturbation_column="perturbation",
        context_key="cell_type",
        control_label="control",
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        precision=args.precision,
        resume=args.resume,
        seed=args.seed,
        timesteps=args.timesteps,
        omega=args.omega,
    )
    _report_generation_quality(
        result, test, perturbation_column="perturbation", control_label="control"
    )


if __name__ == "__main__":
    main()
