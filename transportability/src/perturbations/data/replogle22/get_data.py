"""Preprocess K562-anchored Replogle perturbation dataset pairs."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata.experimental import AnnCollection
from scipy.sparse import csr_matrix
from tqdm import tqdm

from ..preprocess import (
    compute_degs,
    perturbation_targets_in_gene_set,
    write_processed_manifest,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent
K562_CELL_LINE = "K562"


@dataclass(frozen=True)
class PartnerSpec:
    """Configuration for one K562 partner dataset."""

    filename: str
    source_release: str
    source_accession: str
    is_geo: bool = False


@dataclass(frozen=True)
class PreprocessingConfig:
    """Numerical settings shared by all pair preprocessing runs."""

    min_cells_per_pert_per_cell_line: int = 64
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    normalize_target_sum: float = 1e4
    hvg_top_genes: int = 8192


PARTNER_SPECS = {
    "RPE1": PartnerSpec(
        filename="replogle22_RPE1.h5ad",
        source_release="Replogle et al. 2022",
        source_accession="Zenodo 13350497",
    ),
    "Jurkat": PartnerSpec(
        filename="GSE264667_jurkat_raw_singlecell_01.h5ad",
        source_release="Nadig/Replogle et al. 2025",
        source_accession="GSE264667",
        is_geo=True,
    ),
    "HepG2": PartnerSpec(
        filename="GSE264667_hepg2_raw_singlecell_01.h5ad",
        source_release="Nadig/Replogle et al. 2025",
        source_accession="GSE264667",
        is_geo=True,
    ),
}
DEFAULT_PARTNERS = tuple(PARTNER_SPECS)


def run_preprocessing(
    partners: Sequence[str],
    config: PreprocessingConfig,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """
    Generate processed K562 pair datasets.

    Args:
        partners: Partner cell lines to pair with K562.
        config: Shared preprocessing settings.
        data_dir: Directory containing raw inputs and receiving output folders.
    """
    data_dir = Path(data_dir)
    input_paths = validate_input_files(partners, data_dir)

    k562 = read_source_adata(input_paths[K562_CELL_LINE], K562_CELL_LINE)
    for partner_name in partners:
        print(f"\nProcessing K562 + {partner_name}")
        partner_spec = PARTNER_SPECS[partner_name]
        partner = read_source_adata(
            input_paths[partner_name],
            partner_name,
            is_geo=partner_spec.is_geo,
        )
        process_pair(
            adata_k562=k562,
            adata_partner=partner,
            partner_name=partner_name,
            raw_data_paths=[
                input_paths[K562_CELL_LINE],
                input_paths[partner_name],
            ],
            output_dir=data_dir / partner_name,
            config=config,
        )
        del partner
        gc.collect()


def process_pair(
    adata_k562: ad.AnnData,
    adata_partner: ad.AnnData,
    partner_name: str,
    raw_data_paths: Sequence[Path],
    output_dir: Path,
    config: PreprocessingConfig,
) -> None:
    """
    Run the shared preprocessing pipeline for one K562 pair.

    Args:
        adata_k562: Raw K562 single-cell data.
        adata_partner: Raw partner-cell-line single-cell data.
        partner_name: Partner cell-line name.
        raw_data_paths: Source files used for the pair.
        output_dir: Directory for processed artifacts.
        config: Shared preprocessing settings.
    """
    partner_spec = PARTNER_SPECS[partner_name]
    raw_shapes = {
        K562_CELL_LINE: [int(adata_k562.n_obs), int(adata_k562.n_vars)],
        partner_name: [int(adata_partner.n_obs), int(adata_partner.n_vars)],
    }
    duplicate_gene_symbols_removed = int(adata_partner.uns.get("duplicate_gene_symbols_removed", 0))
    collection = AnnCollection(
        [adata_k562, adata_partner],
        join_vars="inner",
        keys=[K562_CELL_LINE.lower(), partner_name.lower()],
        index_unique="-",
    )
    shared_gene_count = int(collection.n_vars)
    counts = perturbation_counts(collection.obs, partner_name)
    minimum_cells = config.min_cells_per_pert_per_cell_line
    valid_perturbations = counts.index[
        (counts[K562_CELL_LINE] >= minimum_cells) & (counts[partner_name] >= minimum_cells)
    ]

    for cell_line in (K562_CELL_LINE, partner_name):
        n_valid = int((counts[cell_line] >= minimum_cells).sum())
        print(f"Number of perturbations with >= {minimum_cells} cells in {cell_line}: {n_valid}")
    print(f"Number retained in both cell lines: {len(valid_perturbations)}")

    obs_mask = collection.obs["perturbation"].isin(valid_perturbations).to_numpy()
    adata = collection[np.flatnonzero(obs_mask), :].to_adata()
    adata.obs["perturbation"] = (
        adata.obs["perturbation"].astype(str).str.replace("_", "+", regex=False)
    )
    adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
    adata.obs["condition"] = adata.obs["perturbation"].copy()
    adata.X = csr_matrix(adata.X)

    sc.pp.filter_cells(adata, min_genes=config.min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=config.min_cells_per_gene)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=config.normalize_target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=config.hvg_top_genes,
        subset=True,
    )

    final_gene_set = set(adata.var_names.astype(str))
    perturbation_labels = adata.obs["perturbation"].astype(str)
    keep_mask = perturbation_labels.map(
        lambda label: perturbation_targets_in_gene_set(label, final_gene_set)
    ).to_numpy()
    if not keep_mask.all():
        adata = adata[keep_mask].copy()
        adata.obs["perturbation"] = adata.obs["perturbation"].cat.remove_unused_categories()
        adata.obs["condition"] = adata.obs["perturbation"].copy()

    add_raw_count_summaries(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    compute_degs(adata, mode="vsrest")
    compute_degs(adata, mode="vscontrol")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = write_pair_artifacts(adata, output_dir)
    manifest_path = output_dir / "processed_manifest.json"
    write_processed_manifest(
        manifest_path,
        adata=adata,
        raw_data_paths=raw_data_paths,
        processed_h5ad_path=artifact_paths["processed_h5ad"],
        genes_csv_path=artifact_paths["genes_csv"],
        artifact_paths={
            key: path for key, path in artifact_paths.items() if key.endswith("_pickle")
        },
        extra_metadata={
            "pair_name": f"K562_essential__{partner_name}_essential",
            "cell_lines": [K562_CELL_LINE, partner_name],
            "source_release": {
                K562_CELL_LINE: "Replogle et al. 2022",
                partner_name: partner_spec.source_release,
            },
            "source_accession": {
                K562_CELL_LINE: "Zenodo 13350497",
                partner_name: partner_spec.source_accession,
            },
            "raw_shapes": raw_shapes,
            "n_genes_shared_before_filtering": shared_gene_count,
            "n_duplicate_partner_gene_symbols_removed": duplicate_gene_symbols_removed,
            "n_perturbations_original": int(counts.shape[0]),
            "n_perturbations_after_min_cell_filter": len(valid_perturbations),
            "min_cells_per_perturbation_per_cell_line": int(
                config.min_cells_per_pert_per_cell_line
            ),
            "cell_filter_min_genes": int(config.min_genes_per_cell),
            "gene_filter_min_cells": int(config.min_cells_per_gene),
            "normalize_target_sum": float(config.normalize_target_sum),
            "hvg_top_genes": int(config.hvg_top_genes),
        },
    )
    print(f"Wrote processed dataset to {artifact_paths['processed_h5ad']}")
    print(f"Wrote processed manifest to {manifest_path}")


def validate_input_files(
    partners: Sequence[str],
    data_dir: Path,
) -> dict[str, Path]:
    """Validate all source files required for a preprocessing run."""
    paths = {
        K562_CELL_LINE: data_dir / "replogle22_K562_essential.h5ad",
        **{partner: data_dir / PARTNER_SPECS[partner].filename for partner in partners},
    }
    for cell_line, path in paths.items():
        is_geo = cell_line != K562_CELL_LINE and PARTNER_SPECS[cell_line].is_geo
        try:
            adata = sc.read_h5ad(path, backed="r")
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid or unreadable {cell_line} input at {path}. "
                "Run `python -m data.replogle22.download_data "
                f"--datasets {cell_line}` to restore it."
            ) from exc
        try:
            validate_source_schema(adata, cell_line=cell_line, is_geo=is_geo)
        finally:
            if adata.file.is_open:
                adata.file.close()
    return paths


def read_source_adata(
    path: Path,
    cell_line: str,
    is_geo: bool = False,
) -> ad.AnnData:
    """Read and standardize one raw source dataset."""
    try:
        adata = sc.read_h5ad(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read {cell_line} input at {path}") from exc
    return standardize_geo_partner(adata, cell_line) if is_geo else adata


def validate_source_schema(
    adata: ad.AnnData,
    cell_line: str,
    is_geo: bool,
) -> None:
    """Validate required observation and gene metadata for one source."""
    required_obs = (
        {"gene", "gem_group", "sgID_AB", "mitopercent"} if is_geo else {"perturbation", "cell_line"}
    )
    required_var = {"gene_name"} if is_geo else set()
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    missing_var = sorted(required_var - set(adata.var.columns))
    if missing_obs or missing_var:
        details = []
        if missing_obs:
            details.append(f"missing obs columns: {missing_obs}")
        if missing_var:
            details.append(f"missing var columns: {missing_var}")
        raise ValueError(f"Invalid {cell_line} schema ({'; '.join(details)})")


def standardize_geo_partner(
    adata: ad.AnnData,
    cell_line: str,
) -> ad.AnnData:
    """Map a GSE264667 partner dataset to the Replogle schema."""
    gene_symbols = adata.var["gene_name"].astype("string")
    valid_symbols = gene_symbols.notna() & gene_symbols.ne("")
    duplicate_symbols = gene_symbols.duplicated(keep=False)
    keep_genes = (valid_symbols & ~duplicate_symbols).to_numpy()
    duplicate_count = int((valid_symbols & duplicate_symbols).sum())

    standardized = adata[:, keep_genes].copy()
    standardized.var["ensembl_id"] = standardized.var_names.astype(str)
    standardized.var_names = standardized.var["gene_name"].astype(str).to_numpy()
    standardized.var_names.name = "gene_name"
    standardized.obs = standardized.obs.rename(
        columns={
            "gem_group": "batch",
            "sgID_AB": "guide_id",
            "mitopercent": "percent_mito",
        }
    )
    standardized.obs["perturbation"] = (
        standardized.obs["gene"].astype(str).replace({"non-targeting": "control"})
    )
    standardized.obs["cell_line"] = pd.Categorical([cell_line] * standardized.n_obs)
    standardized.uns["duplicate_gene_symbols_removed"] = duplicate_count
    return standardized


def perturbation_counts(
    obs: pd.DataFrame,
    partner_cell_line: str,
) -> pd.DataFrame:
    """Count cells for every perturbation and cell line."""
    required_columns = {"perturbation", "cell_line"}
    missing = sorted(required_columns - set(obs.columns))
    if missing:
        raise ValueError(f"Combined data is missing obs columns: {missing}")

    counts = obs.pivot_table(
        index="perturbation",
        columns="cell_line",
        aggfunc="size",
        fill_value=0,
        observed=True,
    )
    required_cell_lines = {K562_CELL_LINE, partner_cell_line}
    missing_cell_lines = sorted(required_cell_lines - set(counts.columns))
    if missing_cell_lines:
        raise ValueError(f"Combined data is missing cell lines: {missing_cell_lines}")
    return counts


def add_raw_count_summaries(adata: ad.AnnData) -> None:
    """Store per-condition raw-count means and variances in ``adata.uns``."""
    conditions = adata.obs["condition"].unique()
    mean_df = pd.DataFrame(index=adata.var_names, columns=conditions)
    dispersion_df = pd.DataFrame(index=adata.var_names, columns=conditions)
    condition_labels = adata.obs["condition"].astype(str).to_numpy()

    for perturbation in tqdm(conditions, desc="Computing count summaries"):
        perturbation_mask = condition_labels == str(perturbation)
        perturbation_counts_matrix = adata.layers["counts"][perturbation_mask].toarray()
        mean_df.loc[:, perturbation] = np.mean(perturbation_counts_matrix, axis=0)
        dispersion_df.loc[:, perturbation] = np.var(
            perturbation_counts_matrix,
            axis=0,
        )

    adata.uns["mean_dict"] = mean_df.to_dict(orient="list")
    adata.uns["disp_dict"] = dispersion_df.to_dict(orient="list")
    adata.uns["mean_disp_dict_genes"] = dispersion_df.index.tolist()


def write_pair_artifacts(
    adata: ad.AnnData,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the processed AnnData, gene metadata, and DEG tables."""
    paths = {
        "processed_h5ad": output_dir / "processed.h5ad",
        "genes_csv": output_dir / "genes.csv.gz",
    }
    for mode, filename_suffix in (("vsrest", "vsrest"), ("vscontrol", "vsctrl")):
        results = adata.uns[f"rank_genes_groups_{mode}"]
        for value in ("names", "scores"):
            path = output_dir / f"{value}_df_{filename_suffix}.pkl"
            pd.DataFrame(results[value]).to_pickle(path)
            paths[f"{value}_df_{mode}_pickle"] = path

    for key in (
        "rank_genes_groups_vsrest",
        "rank_genes_groups_vscontrol",
        "rank_genes_groups",
    ):
        adata.uns.pop(key, None)
    adata.write_h5ad(paths["processed_h5ad"])
    adata.var.to_csv(paths["genes_csv"])
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse preprocessing command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate K562 + RPE1, Jurkat, and HepG2 datasets."
    )
    parser.add_argument(
        "--partners",
        nargs="+",
        choices=DEFAULT_PARTNERS,
        default=list(DEFAULT_PARTNERS),
        help="Partner cell lines to process (default: all three).",
    )
    parser.add_argument(
        "--min-cells-per-pert-per-cell-line",
        type=int,
        default=64,
        help="Minimum cells per perturbation in each cell line (default: 64).",
    )
    parser.add_argument(
        "--min-genes-per-cell",
        type=int,
        default=200,
        help="Minimum detected genes required per cell (default: 200).",
    )
    parser.add_argument(
        "--min-cells-per-gene",
        type=int,
        default=3,
        help="Minimum cells in which a gene must be detected (default: 3).",
    )
    parser.add_argument(
        "--normalize-target-sum",
        type=float,
        default=1e4,
        help="Library-size normalization target sum (default: 10000).",
    )
    parser.add_argument(
        "--hvg-top-genes",
        type=int,
        default=8192,
        help="Number of highly variable genes to retain (default: 8192).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing raw inputs and receiving pair outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Replogle pair preprocessing from the command line."""
    args = parse_args(argv)
    config = PreprocessingConfig(
        min_cells_per_pert_per_cell_line=args.min_cells_per_pert_per_cell_line,
        min_genes_per_cell=args.min_genes_per_cell,
        min_cells_per_gene=args.min_cells_per_gene,
        normalize_target_sum=args.normalize_target_sum,
        hvg_top_genes=args.hvg_top_genes,
    )
    run_preprocessing(
        partners=args.partners,
        config=config,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
