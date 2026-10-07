"""Differential-expression utilities for perturbation metric evaluation."""

# pyright: reportUnknownMemberType=false

import warnings
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from anndata import AnnData


def scanpy_de_table(
    adata: AnnData,
    pert_col: str = "perturbation",
    control_pert: str | int | float = "control",
    key_added: str = "de",
    layer: str | None = None,
) -> pl.DataFrame:
    """Compute a Scanpy DE table comparing each perturbation against a control group."""
    groups = [p for p in adata.obs[pert_col].unique() if p != control_pert]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        sc.tl.rank_genes_groups(
            adata,
            groupby=pert_col,
            groups=groups,
            reference=cast(str, control_pert),
            method="wilcoxon",
            corr_method="benjamini-hochberg",
            n_genes=adata.n_vars,
            key_added=key_added,
            layer=layer,
        )

    frames: list[pd.DataFrame] = []
    for pert in groups:
        df = sc.get.rank_genes_groups_df(adata, group=pert, key=key_added)
        frames.append(
            pd.DataFrame(
                {
                    "target": pert,
                    "feature": df["names"],
                    "fold_change": 2 ** df["logfoldchanges"],
                    "p_value": df["pvals"],
                    "fdr": df["pvals_adj"],
                }
            )
        )

    return pl.from_pandas(pd.concat(frames, ignore_index=True))


def de_table_to_deg_masks(
    de_table: pl.DataFrame,
    gene_names: np.ndarray,
    perturbation_ids: np.ndarray,
    fdr_threshold: float = 0.05,
    not_empty: bool = False,
) -> list[np.ndarray]:
    """
    Convert DE rows into boolean DEG masks aligned to ``gene_names``.

    Args:
        de_table: Differential-expression table containing ``target``, ``feature``,
            ``p_value``, and ``fdr`` columns.
        gene_names: Gene names defining the mask order.
        perturbation_ids: Perturbations to extract.
        fdr_threshold: FDR cutoff for significant DEGs.
        not_empty: Whether to force at least one DEG per perturbation when no genes
            pass ``fdr_threshold``.
    """
    gene_names = np.asarray(gene_names).astype(str)
    gene_to_idx = {gene_name: idx for idx, gene_name in enumerate(gene_names)}
    deg_masks: list[np.ndarray] = []

    for pert in perturbation_ids:
        pert_de = de_table.filter(pl.col("target") == pert).sort(["fdr", "p_value"])
        if pert_de.height == 0:
            raise ValueError(f"No DE results found for perturbation {pert!r}.")

        selected = pert_de.filter(pl.col("fdr") < float(fdr_threshold))
        if selected.height == 0 and not_empty:
            selected = pert_de.head(1)

        mask = np.zeros(gene_names.size, dtype=bool)
        feature_indices = [
            gene_to_idx[feature] for feature in selected.get_column("feature").to_list()
        ]
        mask[np.asarray(feature_indices, dtype=int)] = True
        deg_masks.append(mask)

    return deg_masks


def scanpy_deg_masks(
    adata: AnnData,
    perturbation_ids: np.ndarray,
    fdr_threshold: float = 0.05,
    not_empty: bool = False,
    pert_col: str = "perturbation",
    control_pert: str | int | float = "control",
    key_added: str = "de",
    layer: str | None = None,
) -> list[np.ndarray]:
    """Run Scanpy DE and return DEG masks for the requested perturbation ids."""
    de_table = scanpy_de_table(
        adata=adata,
        pert_col=pert_col,
        control_pert=control_pert,
        key_added=key_added,
        layer=layer,
    )
    return de_table_to_deg_masks(
        de_table=de_table,
        gene_names=np.asarray(adata.var_names),
        perturbation_ids=perturbation_ids,
        fdr_threshold=fdr_threshold,
        not_empty=not_empty,
    )


def des_from_de_tables(
    real_de: pl.DataFrame,
    pred_de: pl.DataFrame,
    fdr_threshold: float = 0.05,
    k: int | None = None,
) -> dict[str, dict[str, dict[str, float] | float]]:
    """
    Calculate all Differential Expression Scores (DES) from DE tables.

    Args:
        real_de: Real DE results with columns ``target``, ``feature``, ``fold_change``,
            ``p_value``, and ``fdr``.
        pred_de: Predicted DE results with the same schema as ``real_de``.
        fdr_threshold: FDR threshold used to determine significant DEGs.
        k: Number of top DEGs to consider per perturbation. If ``None``, use all
            significant DEGs.

    Returns:
        Dictionary keyed by ``recall``, ``precision``, and ``jaccard``. Each metric
        contains:
        - ``per_pert``: perturbation-level DES scores
        - ``overall``: mean DES score across perturbations with non-empty real DEG sets
    """

    def ranked_genes(df: pl.DataFrame) -> dict[str, np.ndarray]:
        if "abs_log2_fold_change" in df.columns:
            x = df
        elif "log2_fold_change" in df.columns:
            x = df.with_columns(pl.col("log2_fold_change").abs().alias("abs_log2_fold_change"))
        else:
            x = df.with_columns(
                (pl.col("fold_change").log() / np.log(2.0)).abs().alias("abs_log2_fold_change")
            )

        x = x.filter(pl.col("fdr") < float(fdr_threshold)).sort(
            ["target", "abs_log2_fold_change"],
            descending=[False, True],
        )

        grouped = x.group_by("target", maintain_order=True).agg(pl.col("feature"))
        return {
            row["target"]: np.asarray(row["feature"], dtype=object)
            for row in grouped.iter_rows(named=True)
        }

    real_ranked = ranked_genes(real_de)
    pred_ranked = ranked_genes(pred_de)

    perts = sorted(real_ranked)
    per_metric_per_pert: dict[str, dict[str, float]] = {
        "recall": {},
        "precision": {},
        "jaccard": {},
    }

    for pert in perts:
        real_genes = real_ranked.get(pert, np.array([], dtype=object))
        pred_genes = pred_ranked.get(pert, np.array([], dtype=object))

        recall_k = len(real_genes) if k is None else min(k, len(real_genes))
        recall_real = set(real_genes[:recall_k])
        recall_pred = set(pred_genes[:recall_k])
        per_metric_per_pert["recall"][pert] = (
            0.0 if recall_k == 0 else len(recall_real & recall_pred) / recall_k
        )

        precision_k = len(pred_genes) if k is None else min(k, len(pred_genes))
        precision_real = set(real_genes[:precision_k])
        precision_pred = set(pred_genes[:precision_k])
        per_metric_per_pert["precision"][pert] = (
            0.0 if precision_k == 0 else len(precision_real & precision_pred) / precision_k
        )

        real_subset = set(real_genes if k is None else real_genes[:k])
        pred_subset = set(pred_genes if k is None else pred_genes[:k])
        union = real_subset | pred_subset
        per_metric_per_pert["jaccard"][pert] = (
            0.0 if len(union) == 0 else len(real_subset & pred_subset) / len(union)
        )

    return {
        metric_name: {
            "per_pert": per_pert,
            "overall": float(np.mean(list(per_pert.values()))) if per_pert else np.nan,
        }
        for metric_name, per_pert in per_metric_per_pert.items()
    }
