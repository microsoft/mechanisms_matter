"""
Data loading utilities for GEARS.

Converts AnnData objects into PyTorch Geometric DataLoaders
compatible with the GEARS model's forward() method.
"""

import anndata
import numpy as np
import torch
from scipy.sparse import issparse  # type: ignore
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def adata_to_pyg_data_list(
    adata: anndata.AnnData,
    perturbation_column: str,
    control_label: str,
    control_label_pert_idx: int,
    node_map_pert: dict[str, int],
    control_expression_pool: np.ndarray,
) -> list[Data]:
    """
    Convert an AnnData object into a list of PyTorch Geometric Data objects.

    Each cell becomes a single Data object with:
        - x: control/baseline gene expression vector (num_genes,)
        - y: observed target expression vector for the cell
        - pert: perturbation condition string
        - pert_idx: list of perturbation gene indices in the GNN graph (supports combo perturbations like "gene_0+gene_1")

    Args:
        adata: AnnData object with log-normalized expression in .X
        perturbation_column: column in adata.obs identifying perturbation conditions
        control_label: label for control cells
        control_label_pert_idx: index for control perturbation in the GNN graph
        node_map_pert: dict mapping perturbation names to graph node indices (-1 for control)
        control_expression_pool: training-control expression profiles used as GEARS inputs

    Returns:
        List of PyTorch Geometric Data objects, one per cell.
    """
    data_list = []

    # GEARS expects x to be a control/baseline profile and y to be the observed target.
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)  # type: ignore[arg-type]
    X = np.asarray(X, dtype=np.float32)
    control_expression_pool = np.asarray(control_expression_pool, dtype=np.float32)
    if control_expression_pool.ndim != 2 or control_expression_pool.shape[1] != X.shape[1]:
        raise ValueError(
            "control_expression_pool must be a 2D array with the same number of genes as adata.X."
        )
    if control_expression_pool.shape[0] == 0:
        raise ValueError("control_expression_pool must contain at least one control cell.")

    perts = adata.obs[perturbation_column].values
    control_indices = np.arange(adata.n_obs, dtype=np.int64) % control_expression_pool.shape[0]

    for i in range(adata.n_obs):
        x = torch.tensor(control_expression_pool[control_indices[i]], dtype=torch.float32)
        y = torch.tensor(X[i], dtype=torch.float32)

        # Get perturbation condition for this cell
        pert = perts[i]

        # Resolve perturbation gene indices (supports combo perturbations like "gene_0+gene_1")
        if pert == control_label:
            pert_idx = [control_label_pert_idx]  # control perturbation index
        else:
            pert_genes = [g for g in str(pert).split("+") if g and g != control_label]
            missing = [g for g in pert_genes if g not in node_map_pert]
            if missing:
                raise KeyError(f"Unknown perturbation target(s) in label {pert!r}: {missing}")
            pert_idx = [node_map_pert[g] for g in pert_genes]

        data = Data(
            x=x,
            y=y,
            pert=pert,
            pert_idx=pert_idx,
        )
        data_list.append(data)  # type: ignore

    return data_list  # type: ignore


def create_dataloader(
    adata: anndata.AnnData,
    perturbation_column: str,
    control_label: str,
    control_label_pert_idx: int,
    node_map_pert: dict[str, int],
    control_expression_pool: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a single DataLoader from a pre-split AnnData object.

    The caller is responsible for splitting the data into train/val/test
    before calling this function. Pass the appropriate split's AnnData.

    Args:
        adata: AnnData object for a single split (train, val, or test)
            with log-normalized expression in .X
        perturbation_column: column in adata.obs identifying perturbation conditions
        control_label: label for control cells
        control_label_pert_idx: index for control perturbation (-1 by convention)
        node_map_pert: dict mapping perturbation names to graph node indices
        control_expression_pool: training-control expression profiles used as GEARS inputs
        batch_size: batch size for DataLoader
        shuffle: whether to shuffle the data (True for train, False for val/test)
        drop_last: whether to drop the final incomplete batch

    Returns:
        PyTorch Geometric DataLoader.
    """
    if issparse(adata.X):  # type: ignore
        # For sparse matrices, only check non-zero entries
        data = adata.X.data  #  # type: ignore
        is_all_integer = np.allclose(data, np.round(data))  # type: ignore
    else:
        is_all_integer = np.allclose(adata.X, np.round(adata.X))  # type: ignore
    assert not is_all_integer, (
        "adata.X contains only integer values — this looks like raw counts. "
        "Expected float values after log1p normalization. "
        "Run sc.pp.normalize_total(adata) and sc.pp.log1p(adata) first."
    )
    data_list = adata_to_pyg_data_list(
        adata=adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        control_label_pert_idx=control_label_pert_idx,
        node_map_pert=node_map_pert,
        control_expression_pool=control_expression_pool,
    )

    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
