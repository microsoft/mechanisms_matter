"""Generate synthetic perturbation data and save UMAP plots by perturbation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from umap import UMAP

from perturbations.analyses.plot_utils import apply_paper_plot_style
from perturbations.data.dgp.causalDGP import causalDGP
from perturbations.metrics.reconstruction.distance_util import estimate_mmd_gamma
from perturbations.metrics.reconstruction.vendi_score import (
    estimate_vendi_outer_sigma_squared,
    vendi_score,
)
from perturbations.util.anndata_util import fit_control_incremental_pca

matplotlib.use("Agg")


CONTROL_LABEL = "control"
CELL_LINE_COLOR_MAP = {0: "#1f77b4", 1: "#d62728"}
CELL_LINE_MARKER_MAP = {0: "o", 1: "^"}
UMAP_FIG_SIZE = (9.0, 7.0)
UMAP_SCATTER_SIZE = 20
UMAP_SCATTER_ALPHA = 0.82
SIDE_LEGEND_X = 1.02


def _perturbation_color_map(perturbation: np.ndarray) -> dict[str, object]:
    import matplotlib.pyplot as plt

    perturbation_values = np.unique(perturbation).tolist()
    perturbation_order = [
        CONTROL_LABEL,
        *sorted(label for label in perturbation_values if label != CONTROL_LABEL),
    ]
    cmap = plt.get_cmap("tab10")
    color_map = {CONTROL_LABEL: "#7f7f7f"}
    for i, label in enumerate(perturbation_order[1:]):
        color_map[label] = cmap(i % cmap.N)
    return color_map


def _plot_umap(
    embedding: np.ndarray,
    cell_line: np.ndarray,
    output_path: Path,
    perturbation: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_paper_plot_style()
    fig, ax = plt.subplots(figsize=UMAP_FIG_SIZE)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if perturbation is None:
        point_colors = np.asarray(
            [CELL_LINE_COLOR_MAP.get(int(ct), "#7f7f7f") for ct in cell_line],
            dtype=object,
        )
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=point_colors.tolist(),
            marker="o",
            s=UMAP_SCATTER_SIZE,
            alpha=UMAP_SCATTER_ALPHA,
            linewidths=0.0,
            rasterized=True,
        )
        ax.legend(
            handles=_cell_line_legend_handles(CELL_LINE_COLOR_MAP),
            title="Cell Line",
            loc="center left",
            bbox_to_anchor=(SIDE_LEGEND_X, 0.5),
            borderaxespad=0.0,
        )
    else:
        perturbation_color_map = _perturbation_color_map(perturbation)
        present_labels = [
            label for label in perturbation_color_map if np.any(perturbation == label)
        ]
        for line in sorted(np.unique(cell_line).tolist()):
            line_marker = CELL_LINE_MARKER_MAP.get(int(line), "o")
            for label in present_labels:
                idx = (cell_line == line) & (perturbation == label)
                if not np.any(idx):
                    continue
                ax.scatter(
                    embedding[idx, 0],
                    embedding[idx, 1],
                    c=[perturbation_color_map[label]],
                    marker=line_marker,
                    s=UMAP_SCATTER_SIZE,
                    alpha=UMAP_SCATTER_ALPHA,
                    linewidths=0.0,
                    rasterized=True,
                )

        cell_line_legend = ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker=CELL_LINE_MARKER_MAP.get(int(line), "o"),
                    linestyle="None",
                    markerfacecolor="#4c4c4c",
                    markeredgecolor="none",
                    markersize=9.5,
                    label=f"Line {line}",
                )
                for line in sorted(np.unique(cell_line).tolist())
            ],
            title="Cell Line",
            loc="upper left",
            bbox_to_anchor=(SIDE_LEGEND_X, 1.0),
            borderaxespad=0.0,
        )
        ax.add_artist(cell_line_legend)
        perturbation_handles = [
            Patch(
                facecolor=perturbation_color_map[label],
                edgecolor="none",
                label=str(label),
            )
            for label in present_labels
        ]
        ax.legend(
            handles=perturbation_handles,
            title="Perturbation",
            loc="upper left",
            bbox_to_anchor=(SIDE_LEGEND_X, 0.58),
            borderaxespad=0.0,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(False)
    sns.despine(ax=ax)

    fig.subplots_adjust(left=0.12, bottom=0.14, top=0.96, right=0.74)
    fig.savefig(output_path)
    plt.close(fig)


def _cell_line_legend_handles(color_map: dict[int, str]) -> list[object]:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[0],
            markeredgecolor="none",
            markersize=9.5,
            label="Line 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[1],
            markeredgecolor="none",
            markersize=9.5,
            label="Line 1",
        ),
    ]


def _compute_sparsity(matrix) -> tuple[int, int, float]:
    total_entries = int(matrix.shape[0] * matrix.shape[1])
    if sparse.issparse(matrix):
        nonzero_entries = int(matrix.nnz)
    else:
        nonzero_entries = int(np.count_nonzero(np.asarray(matrix)))
    sparsity = 1.0 - (nonzero_entries / total_entries) if total_entries > 0 else float("nan")
    return total_entries, nonzero_entries, sparsity


def main() -> None:
    """Generate synthetic data and produce per-perturbation UMAP figures."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic causal DGP data and plot UMAP by cell line per perturbation."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/other_plots",
        help="Directory for generated figures.",
    )
    parser.add_argument("--G", type=int, default=128, help="Number of genes.")
    parser.add_argument("--N0", type=int, default=1024, help="Number of control cells.")
    parser.add_argument("--Nk", type=int, default=1024, help="Number of cells per perturbation.")
    parser.add_argument("--P", type=int, default=5, help="Number of perturbations.")
    parser.add_argument(
        "--mu_l", type=float, default=1.0, help="Mean of log library size for the synthetic data."
    )
    parser.add_argument(
        "--diversity-type",
        type=str,
        default="both",
        choices=["A", "b", "both", "none"],
        help="Type of diversity.",
    )
    parser.add_argument(
        "--swap-fraction",
        type=float,
        default=0.5,
        help="Fraction of A edges rewired to create A_alter.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.15, help="UMAP min_dist.")
    parser.add_argument(
        "--pca-n-components",
        type=int,
        default=50,
        help="Number of PCA components used before fitting UMAP.",
    )
    parser.add_argument(
        "--normalized-layer-key",
        type=str,
        default="normalized_log1p",
        help="Layer key used as UMAP input (falls back to .X if missing).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_params_df = pd.read_csv(
        "results/synthetic_simulations/parameter_estimation/all_fitted_params.csv",
        index_col=0,
    )
    all_theta = all_params_df["n"].values
    gene_names = all_params_df.index.to_numpy(dtype=str)

    adata, _all_affected_masks = causalDGP(
        G=args.G,
        N0=args.N0,
        Nk=args.Nk,
        P=args.P,
        mu_l=args.mu_l,
        all_theta=all_theta,
        gene_names=gene_names,
        swap_fraction=args.swap_fraction,
        seed=args.seed,
        mask_method="Erdos-Renyi",
        diversity_type=args.diversity_type,
        normalize=True,
        normalized_layer_key=args.normalized_layer_key,
        visualize=True,
        verbose=True,
        output_dir=str(output_dir),
    )
    vendi_pca_model = fit_control_incremental_pca(
        data_obj=adata,
        layer_key=args.normalized_layer_key,
        control_label=CONTROL_LABEL,
        obs_key="perturbation",
        data_name="adata",
    )
    vendi_gamma = estimate_mmd_gamma(
        obs=adata,
        layer_obs=args.normalized_layer_key,
        control_label=CONTROL_LABEL,
        seed=args.seed,
        pca_model=vendi_pca_model,
    )
    vendi_outer_sigma_squared = estimate_vendi_outer_sigma_squared(
        ac=adata,
        gamma=vendi_gamma,
        pca_model=vendi_pca_model,
        layer_key=args.normalized_layer_key,
        control_label=CONTROL_LABEL,
        random_state=args.seed,
    )
    vendi = vendi_score(
        ac=adata,
        layer_key=args.normalized_layer_key,
        control_label=CONTROL_LABEL,
        gamma=vendi_gamma,
        pca_model=vendi_pca_model,
        outer_sigma_squared=vendi_outer_sigma_squared,
    )
    print(f"Vendi score for the dataset: {vendi}")

    # sparsity
    count_matrix = adata.X
    total_entries, nonzero_entries, sparsity = _compute_sparsity(count_matrix)
    print(f"Total entries: {total_entries}")
    print(f"Non-zero entries: {nonzero_entries}")
    print(f"Sparsity of the dataset: {sparsity:.4f}")

    required_obs = {"cell_line", "perturbation"}
    missing_obs = required_obs.difference(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"Missing required obs columns: {sorted(missing_obs)}")

    if args.normalized_layer_key in adata.layers:
        print(f"Using layer '{args.normalized_layer_key}' for UMAP input.")
        X = adata.layers[args.normalized_layer_key]
    else:
        print(
            f"Layer '{args.normalized_layer_key}' not found; falling back to adata.X for UMAP input."
        )
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()

    perturbation = adata.obs["perturbation"].astype(str).to_numpy(copy=False)
    control_idx = perturbation == CONTROL_LABEL
    pca_model = PCA(n_components=args.pca_n_components, random_state=args.seed)
    pca_embedding = pca_model.fit_transform(X)

    reducer = UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    )
    embedding = reducer.fit_transform(pca_embedding)

    cell_line = adata.obs["cell_line"].to_numpy(dtype=np.int32, copy=False)
    control_fig_path = output_dir / "umap_cell_line_control.svg"
    _plot_umap(
        embedding=embedding[control_idx],
        cell_line=cell_line[control_idx],
        output_path=control_fig_path,
    )
    print(f"Saved UMAP plot to: {control_fig_path}")

    all_cells_fig_path = output_dir / "umap_all_cells_cell_line_perturbation.svg"
    _plot_umap(
        embedding=embedding,
        cell_line=cell_line,
        perturbation=perturbation,
        output_path=all_cells_fig_path,
    )
    print(f"Saved UMAP plot to: {all_cells_fig_path}")

    perturbation_values = sorted(
        label for label in np.unique(perturbation).tolist() if label != CONTROL_LABEL
    )

    if len(perturbation_values) == 0:
        raise ValueError("No non-control perturbation cells found to plot.")

    for p in perturbation_values:
        idx = perturbation == str(p)
        fig_path = output_dir / f"umap_cell_line_perturbation_{p}.svg"
        _plot_umap(
            embedding=embedding[idx],
            cell_line=cell_line[idx],
            output_path=fig_path,
        )
        print(f"Saved UMAP plot to: {fig_path}")


if __name__ == "__main__":
    main()
