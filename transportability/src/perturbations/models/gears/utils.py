"""Utility functions for GEARS model."""
# Adapted from https://github.com/snap-stanford/GEARS/blob/f374e43e197b295016d80395d7a54ddb81cc6769/gears/utils.py

import pickle
from pathlib import Path

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm


def project_to_normalized_log1p(
    matrix: np.ndarray,
    target_sum: float = 1e4,
) -> np.ndarray:
    """
    Project predictions onto a valid normalized+log1p expression manifold.

    GEARS is trained against normalized/log1p targets, but its decoder is an
    unconstrained regressor, so it can still emit negative or non-finite values.

    This helper therefore treats model output as coordinates in log1p space and
    projects them back onto the subset of values that correspond to a valid
    non-negative normalized expression profile with a fixed library-size target.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={arr.shape}.")
    if target_sum <= 0:
        raise ValueError(f"target_sum must be > 0, got {target_sum}.")

    projected = arr.copy()
    # NaN/inf have no meaningful interpretation as expression values, so map
    # them to zero before projecting back into expression space.
    projected[~np.isfinite(projected)] = 0.0
    # Negative log1p values would imply negative expression after `expm1`, which
    # is not a valid count-derived signal for Scanpy's DE routines.
    np.maximum(projected, 0.0, out=projected)

    # Move back to linear expression space, where row-wise library-size
    # normalization has a clear interpretation.
    linear = np.expm1(projected)
    # Numerical noise or earlier invalid values can still leave tiny negatives.
    # Clamp again so the renormalized profile is strictly non-negative.
    np.maximum(linear, 0.0, out=linear)

    row_sums = linear.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums[:, 0] > 0.0
    if np.any(nonzero_rows):
        # Enforce the same per-cell target sum used for normalized expression so
        # the returned values match the semantics of the `normalized_log1p` layer.
        linear[nonzero_rows] *= float(target_sum) / row_sums[nonzero_rows]

    # Return to the representation used throughout the rest of the pipeline.
    np.log1p(linear, out=linear)
    return linear.astype(np.float32, copy=False)


class GeneSimNetwork:
    """
    GeneSimNetwork class.

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """

    def __init__(self, edge_list: pd.DataFrame, gene_list: list[str], node_map: dict[str, int]):
        """Initialize GeneSimNetwork class."""
        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(  # type: ignore
            df=self.edge_list,
            source="source",
            target="target",
            edge_attr=["importance"],
            create_using=nx.DiGraph,
        )
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():  # type: ignore
                self.G.add_node(n)  # type: ignore

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in self.G.edges]  # type: ignore
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        # self.edge_weight = torch.Tensor(self.edge_list['importance'].values)

        edge_attr = nx.get_edge_attributes(self.G, "importance")  # type: ignore
        importance = np.array([edge_attr[e] for e in self.G.edges])  # type: ignore
        self.edge_weight = torch.Tensor(importance)


def np_pearson_cor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.

    Returns:
    -------
    np.ndarray
        Pearson correlation coefficient matrix, bounded to [-1, 1].
    """
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))  # type: ignore
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def get_coexpression_network_from_train(
    adata: anndata.AnnData,
    threshold: float,
    k: int,
    data_path: str,
    data_name: str,
    split: str,
    seed: int,
    train_gene_set_size: int,
    set2conditions: dict[str, list[str]],
    control_label: str,
    perturbation_column: str,
) -> pd.DataFrame:
    """
    Infer co-expression network from training data.

    Args:
        adata (anndata.AnnData): anndata object
        threshold (float): threshold for co-expression
        k (int): number of edges to keep
        data_path (str): path to data
        data_name (str): name of dataset
        split (str): split of dataset
        seed (int): seed for random number generator
        train_gene_set_size (int): size of training gene set
        set2conditions (dict): dictionary of perturbations to conditions
        control_label (str): label for control perturbation
        perturbation_column (str): column in adata.obs identifying perturbation conditions
    """
    fname = (
        Path(data_path)
        / data_name
        / (
            split
            + "_"
            + str(seed)
            + "_"
            + str(train_gene_set_size)
            + "_"
            + str(threshold)
            + "_"
            + str(k)
            + "_co_expression_network.csv"
        )
    )
    print(f"Looking for co-expression network at {fname}...")

    if fname.exists():
        return pd.read_csv(fname)
    else:
        fname.parent.mkdir(parents=True, exist_ok=True)
        gene_list: list[str] = [str(f) for f in adata.var_names.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list, strict=True))
        X = adata.X  # type: ignore
        train_perts = set2conditions["train"]
        X_tr = X[  # type: ignore
            np.isin(adata.obs[perturbation_column], [i for i in train_perts if i == control_label])
        ]

        X_tr = X_tr.toarray()  # type: ignore
        out = np_pearson_cor(X_tr, X_tr)  # type: ignore
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1) :]
        out_sort_val = np.sort(out)[:, -(k + 1) :]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))  # type: ignore

        df_g = [i for i in df_g if i[2] > threshold]  # type: ignore
        df_co_expression = pd.DataFrame(df_g).rename(  # type: ignore
            columns={0: "source", 1: "target", 2: "importance"}
        )
        df_co_expression.to_csv(fname, index=False)  # type: ignore
        return df_co_expression  # type: ignore


def uncertainty_loss_fct(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    control_label: str,
    y: torch.Tensor,
    perts: list[str],
    reg: float = 0.1,
    ctrl: torch.Tensor | None = None,
    direction_lambda: float = 1e-3,
    dict_filter: dict[str, list[int]] | None = None,
):
    """
    Uncertainty loss function.

    Args:
        pred (torch.tensor): predicted values
        logvar (torch.tensor): log variance
        y (torch.tensor): true values
        control_label (str): label for control perturbation
        perts (list): list of perturbations
        reg (float): regularization parameter
        ctrl (torch.tensor): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    if y.dim() == 1:
        # print(f"Warning: y is 1-dimensional, reshaping to match pred., y shape is {y.shape}")
        y = y.reshape(pred.shape)
    gamma = 2
    perts_array = np.array(perts)
    losses = pred.new_tensor(0.0)
    for p in set(perts):
        if p != control_label:
            retain_idx = dict_filter[p]  # type: ignore
            pred_p = pred[np.where(perts_array == p)[0]][:, retain_idx]
            y_p = y[np.where(perts_array == p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts_array == p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts_array == p)[0]]
            y_p = y[np.where(perts_array == p)[0]]
            logvar_p = logvar[np.where(perts_array == p)[0]]

        # uncertainty based loss
        losses = losses + (
            torch.sum(
                (pred_p - y_p) ** (2 + gamma)
                + reg * torch.exp(-logvar_p) * (pred_p - y_p) ** (2 + gamma)
            )
            / pred_p.shape[0]
            / pred_p.shape[1]
        )

        # direction loss
        if p != control_label:
            losses = losses + (
                torch.sum(
                    direction_lambda
                    * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))  # type: ignore
                    ** 2
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )
        else:
            losses = losses + (
                torch.sum(
                    direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl)) ** 2  # type: ignore
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )

    return losses / (len(set(perts)))


def loss_fct(
    pred: torch.Tensor,
    y: torch.Tensor,
    control_label: str,
    perts: list[str],
    ctrl: torch.Tensor | None = None,
    direction_lambda: float = 1e-3,
    dict_filter: dict[str, list[int]] | None = None,
):
    """
    Main MSE Loss function, includes direction loss.

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        control_label (str): label for control perturbation
        perts (list): list of perturbations
        ctrl (torch.tensor): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    if y.dim() == 1:
        # print(f"Warning: y is 1-dimensional, reshaping to match pred., y shape is {y.shape}")
        y = y.reshape(pred.shape)
    gamma = 2
    perts_array = np.array(perts)
    losses = pred.new_tensor(0.0)

    for p in set(perts):
        pert_idx = np.where(perts_array == p)[0]

        # during training, we remove the all zero genes into calculation of loss.
        # this gives a cleaner direction loss. empirically, the performance stays the same.
        if p != control_label:
            retain_idx = dict_filter[p]  # type: ignore
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
        losses = (
            losses + torch.sum((pred_p - y_p) ** (2 + gamma)) / pred_p.shape[0] / pred_p.shape[1]
        )

        ## direction loss
        if p != control_label:
            losses = (
                losses
                + torch.sum(
                    direction_lambda
                    * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))  # type: ignore
                    ** 2
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )
        else:
            losses = (
                losses
                + torch.sum(
                    direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl)) ** 2  # type: ignore
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )
    return losses / (len(set(perts)))


def get_non_zeros_gene_idx(
    adata: anndata.AnnData,
    perturbation_column: str = "perturbation",
    control_label: str = "control",
) -> dict[str, np.ndarray]:
    """
    Compute indices of genes with non-zero mean expression per perturbation condition.

    For each non-control perturbation, computes the mean expression across all cells
    with that perturbation and returns the sorted indices of genes with non-zero mean.

    Args:
        adata: AnnData object with expression data in .X
        perturbation_column: column in adata.obs identifying perturbation conditions
        control_label: label for control cells (excluded from output)

    Returns:
        Dictionary mapping perturbation name to sorted array of non-zero gene indices.
    """
    non_zeros_gene_idx = {}

    for cond in adata.obs[perturbation_column].unique():
        if cond == control_label:
            continue

        # Mean expression across cells with this perturbation
        mask = (adata.obs[perturbation_column] == cond).values
        mean_expr = np.asarray(adata.X[mask].mean(axis=0)).flatten()  # type: ignore

        # Indices of genes with non-zero mean expression
        non_zeros_gene_idx[cond] = np.sort(np.nonzero(mean_expr)[0])  # type: ignore

    return non_zeros_gene_idx  # type: ignore


"""
Gene Ontology and co-expression graph utilities for GEARS.
"""


def dataverse_download(url: str, save_path: str) -> None:
    """
    Dataverse download helper with progress bar.

    Args:
        url (str): the url of the dataset
        save_path (str): the path to save the dataset
    """
    if Path(save_path).exists():
        print("Found local copy...")
    else:
        print("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with Path(save_path).open("wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def get_go_auto(gene_list: list[str], data_path: str, data_name: str) -> pd.DataFrame:
    """
    Get gene ontology data.

    Args:
        gene_list (list): list of gene names
        data_path (str): the path to save the extracted dataset
        data_name (str): the name of the dataset

    Returns:
        df_edge_list (pd.DataFrame): gene ontology edge list
    """
    go_path = Path(data_path) / data_name / "go.csv"

    if go_path.exists():
        return pd.read_csv(go_path)
    else:
        ## download gene2go.pkl
        # create directory
        go_path.parent.mkdir(parents=True, exist_ok=True)
        gene2go_path = Path(data_path) / "gene2go.pkl"
        if not gene2go_path.exists():
            server_path = "https://dataverse.harvard.edu/api/access/datafile/6153417"
            dataverse_download(server_path, str(gene2go_path))
        with gene2go_path.open("rb") as f:
            gene2go = pickle.load(f)

        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list: list[tuple[str, str, float]] = []
        for g1 in tqdm(gene2go.keys()):
            for g2 in gene2go.keys():
                edge_list.append(
                    (
                        g1,
                        g2,
                        len(np.intersect1d(gene2go[g1], gene2go[g2]))
                        / len(np.union1d(gene2go[g1], gene2go[g2])),
                    )
                )

        further_filter = [i for i in edge_list if i[2] > 0.1]
        df_edge_list = pd.DataFrame(further_filter).rename(  # type: ignore
            columns={0: "gene1", 1: "gene2", 2: "score"}
        )

        df_edge_list = df_edge_list.rename(  # type: ignore
            columns={"gene1": "source", "gene2": "target", "score": "importance"}
        )
        df_edge_list.to_csv(go_path, index=False)  # type: ignore
        return df_edge_list  # type: ignore
