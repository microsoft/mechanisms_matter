"""
GEARS trainer and wrapper for gene expression perturbation prediction.

This module provides the GEARS class for training and evaluating a graph neural network
model that predicts gene expression changes in response to genetic perturbations.
It includes model initialization, training loops, and prediction functionality.

Adapted from: https://github.com/snap-stanford/GEARS/tree/f374e43e197b295016d80395d7a54ddb81cc6769/gears
"""

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import (
    csr_matrix,
    issparse,  # type: ignore
)
from scipy.stats import pearsonr  # type: ignore
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from ...util.anndata_util import materialize_adata
from ..test_synthentic_data import generate_synthetic_perturbation_data
from .data_loader import create_dataloader
from .model import GEARS_Model
from .utils import (
    GeneSimNetwork,
    get_coexpression_network_from_train,
    get_go_auto,
    get_non_zeros_gene_idx,
    loss_fct,
    project_to_normalized_log1p,
    uncertainty_loss_fct,
)

_seed = 42
torch.manual_seed(_seed)  # type: ignore
np.random.seed(_seed)  # noqa
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)


def check_last_singleton_batch(
    n_obs: int,
    batch_size: int,
    model_name: str,
) -> bool:
    """Return True when the final training minibatch would contain exactly one sample."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if n_obs <= 0:
        raise ValueError(f"{model_name} received no training cells.")
    if n_obs == 1:
        raise ValueError(
            f"{model_name} received only 1 training cell. "
            "Dropping a singleton residual minibatch would remove the entire training set."
        )
    return n_obs > batch_size and (n_obs % batch_size == 1)


class GEARS:
    """
    GEARS trainer/wrapper.

    Refactored to:
    - build model from model.py (GEARS_Model)
    - build graph tensors from utils.py (GeneSimNetwork, get_coexpression_network_from_train)
    - optionally train/evaluate on synthetic data from utils.py
    """

    def __init__(
        self,
        pert_data: Any,
        device: str = "cuda",
        weight_bias_track: bool = False,
        proj_name: str = "GEARS",
        exp_name: str = "GEARS",
    ) -> None:
        """
        Initialize GEARS trainer with perturbation data and configuration.

        Args:
            pert_data: Perturbation data object containing dataloader, adata, and metadata.
            device: Device to run computations on (default: "cuda").
            weight_bias_track: Whether to track experiments with Weights & Biases (default: False).
            proj_name: Weights & Biases project name (default: "GEARS").
            exp_name: Weights & Biases experiment name (default: "GEARS").
            synthetic: Whether to use synthetic data (default: False).
        """
        self.weight_bias_track = weight_bias_track

        if self.weight_bias_track:
            import wandb

            wandb.init(project=proj_name, name=exp_name)
            self.wandb = wandb
        else:
            self.wandb = None

        self.device = device
        self.config: dict[str, Any] | None = None

        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.perturbation_column = pert_data.perturbation_column
        self.control_label = pert_data.control_label
        self.num_genes = len(self.gene_list)
        # Fo constructing the embeddings to be same as num_genes since we use gene embeddings to represent perturbations, but we also need to know the number of unique perturbations for the pert_emb module
        self.num_perts = self.num_genes + 1  # +1 for control perturbation which is not in gene list

        self.control_label_pert_idx = self.node_map_pert[self.control_label]
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred: dict[str, Any] = {}
        self.saved_logvar_sum: dict[str, Any] = {}

        self.ctrl_expression = (
            torch.tensor(
                np.mean(
                    self.adata.X[
                        (self.adata.obs[self.perturbation_column] == self.control_label).values
                    ],
                    axis=0,
                )
            )
            .reshape(-1)
            .to(self.device)
        )
        pert_full_id2pert = {p: p for p in self.adata.obs[self.perturbation_column].unique()}
        self.dict_filter = {
            pert_full_id2pert[i]: j
            for i, j in self.adata.uns["non_zeros_gene_idx"].items()
            if i in pert_full_id2pert
        }
        self.ctrl_adata = self.adata[
            (self.adata.obs[self.perturbation_column] == self.control_label).values
        ]

        gene_dict = {g: i for i, g in enumerate(self.gene_list)}
        self.pert2gene = {
            p: gene_dict[pert] for p, pert in enumerate(self.pert_list) if pert in self.gene_list
        }

        self.model: GEARS_Model | None = None
        self.best_model: GEARS_Model | None = None

    def initialize_model(
        self,
        *,
        hidden_size: int = 64,
        num_go_gnn_layers: int = 1,
        num_gene_gnn_layers: int = 1,
        decoder_hidden_size: int = 16,
        uncertainty: bool = False,
        uncertainty_reg: float = 1.0,
        direction_lambda: float = 1.0,
        no_perturb: bool = False,
        G_coexpress: torch.Tensor | None = None,
        G_coexpress_weight: torch.Tensor | None = None,
        coexpress_threshold: float = 0.4,
        num_similar_genes_co_express_graph: int = 20,
        save_interval: int = 5,
    ) -> None:
        """Initialize GEARS_Model from model.py using graph builders in utils.py."""
        edge_list = get_coexpression_network_from_train(
            adata=self.adata,
            data_path=self.data_path,
            split=self.split,
            seed=self.seed,
            train_gene_set_size=self.train_gene_set_size,
            set2conditions=self.set2conditions,
            threshold=coexpress_threshold,
            k=num_similar_genes_co_express_graph,
            data_name=self.dataset_name,
            control_label=self.control_label,
            perturbation_column=self.perturbation_column,
        )
        gene_graph = GeneSimNetwork(
            edge_list=edge_list,
            gene_list=self.gene_list,
            node_map=self.node_map,
        )
        if G_coexpress is None or G_coexpress_weight is None:
            G_coexpress = gene_graph.edge_index
            G_coexpress_weight = gene_graph.edge_weight
        if (
            self.default_pert_graph["edge_index"] is None
            or self.default_pert_graph["edge_weight"] is None
        ):
            print("No Gene ontology graph provided, using co-expression graph instead")
            G_sim = G_coexpress
            G_sim_weight = G_coexpress_weight
        else:
            G_sim = self.default_pert_graph["edge_index"]
            G_sim_weight = self.default_pert_graph["edge_weight"]

        self.config = {
            "device": self.device,
            "num_genes": self.num_genes,
            "num_perts": self.num_perts,
            "hidden_size": hidden_size,
            "num_go_gnn_layers": num_go_gnn_layers,
            "num_gene_gnn_layers": num_gene_gnn_layers,
            "decoder_hidden_size": decoder_hidden_size,
            "uncertainty": uncertainty,
            "uncertainty_reg": uncertainty_reg,
            "direction_lambda": direction_lambda,
            "no_perturb": no_perturb,
            "G_coexpress": G_coexpress.to(self.device),
            "G_coexpress_weight": G_coexpress_weight.to(self.device),
            "G_sim": G_sim.to(self.device),
            "G_sim_weight": G_sim_weight.to(self.device),
            "pert2gene": self.pert2gene,
            "synthetic": False,
            "control_label_pert_idx": self.control_label_pert_idx,
            "save_interval": save_interval,
        }

        self.model = GEARS_Model(self.config).to(self.device)

    def train(self, epochs: int = 20, lr: float = 1e-3, weight_decay: float = 5e-4) -> None:
        """
        Train model for either real GEARS path or synthetic path.

        Train model for either:
        - real GEARS path (existing metrics + analyses), or
        - synthetic path (MSE only).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        if self.config is None:
            raise RuntimeError("Config missing. Call initialize_model() first.")

        train_loader = self.dataloader["train_loader"]
        val_loader = self.dataloader["val_loader"]

        self.model = self.model.to(self.device)
        self.best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print("Start Training...")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss: float = 0.0

            for _, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()

                if self.config.get("synthetic", False):
                    # Synthetic mode: direct MSE to batch.y
                    y = batch.y
                    pred = self.model(batch)
                    if isinstance(pred, tuple):  # uncertainty head enabled
                        pred: torch.Tensor = pred[0]
                    loss = F.mse_loss(pred, y)

                else:
                    y = batch.y
                    if self.config["uncertainty"]:
                        # print("Using uncertainty loss function")
                        pred, logvar = self.model(batch)
                        loss: torch.Tensor = uncertainty_loss_fct(
                            pred=pred,
                            logvar=logvar,
                            y=y,
                            perts=batch.pert,
                            reg=self.config["uncertainty_reg"],
                            ctrl=self.ctrl_expression,
                            dict_filter=self.dict_filter,
                            direction_lambda=self.config["direction_lambda"],
                            control_label=self.control_label,
                        )
                    else:
                        # print("Using standard loss function without uncertainty")
                        pred = self.model(batch)
                        loss = loss_fct(
                            pred=pred,
                            y=y,
                            perts=batch.pert,
                            ctrl=self.ctrl_expression,
                            dict_filter=self.dict_filter,
                            direction_lambda=self.config["direction_lambda"],
                            control_label=self.control_label,
                        )
                    epoch_loss += loss.item()

                loss.backward()  # type: ignore
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()  # type: ignore

                if self.wandb:
                    self.wandb.log({"training_loss": float(loss.item())})

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}  Train Loss: {epoch_loss:.4f}")

            scheduler.step()

            # Epoch-level evaluation using loss
            print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f} ")

            if self.wandb:
                self.wandb.log(
                    {
                        "train_loss_epoch": epoch_loss,
                        "epoch": epoch + 1,
                    }
                )

            if epoch % self.config["save_interval"] == 0 and val_loader is not None:
                valid_pred = self.predict(val_loader, self.config["direction_lambda"], self.model)
                if valid_pred["loss"] < min_val:
                    min_val = valid_pred["loss"]
                    print(
                        f"New best model found at epoch {epoch + 1} with val MSE: {valid_pred['loss']:.4f}"
                    )
                    self.best_model = deepcopy(self.model)
                else:
                    print(
                        f"Epoch {epoch + 1} no improvement in validation loss (current: {valid_pred['loss']:.4f}, best: {min_val:.4f})"
                    )

        print("Done!")

    def predict(
        self, dataloader: DataLoader, direction_lambda: float, best_model: GEARS_Model
    ) -> dict[str, Any]:
        """Predict gene expression for a given batch."""
        best_model.eval()

        all_preds = []
        all_truths = []
        all_perts: list[str] = []
        all_logvars = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                out = best_model(batch)
                if isinstance(out, tuple):
                    pred, logvar = out  # type: ignore
                else:
                    pred = out
                    logvar = torch.zeros_like(pred)  # dummy logvar if uncertainty not enabled
                y = batch.y
                if y.dim() == 1:
                    y = y.reshape(pred.shape)  # type: ignore
                if logvar.dim() == 1:  # type: ignore
                    logvar = logvar.reshape(pred.shape)  # type: ignore
                all_preds.append(pred)  # type: ignore
                all_truths.append(y)  # type: ignore
                all_perts.extend(batch.pert)
                all_logvars.append(logvar)  # type: ignore
            all_preds = torch.cat(all_preds, dim=0)  # type: ignore
            all_truths = torch.cat(all_truths, dim=0)  # type: ignore
            all_logvars = torch.cat(all_logvars, dim=0)  # type: ignore
            if self.config and self.config.get("uncertainty", False):
                all_loss = uncertainty_loss_fct(
                    pred=all_preds,
                    logvar=all_logvars,
                    y=all_truths,
                    perts=all_perts,
                    reg=self.config["uncertainty_reg"],
                    ctrl=self.ctrl_expression,  # TODO Note using control expression from training data for validation loss calculation, which may be slightly different from actual control expression in val data but should be close if train and val are similarly distributed
                    dict_filter=self.dict_filter,
                    direction_lambda=direction_lambda,
                    control_label=self.control_label,
                )
            else:
                all_loss = loss_fct(
                    pred=all_preds,
                    y=all_truths,
                    perts=all_perts,
                    ctrl=self.ctrl_expression,  # TODO Note using control expression from training data for validation loss calculation, which may be slightly different from actual control expression in val data but should be close if train and val are similarly distributed
                    dict_filter=self.dict_filter,
                    direction_lambda=direction_lambda,
                    control_label=self.control_label,
                )
        return {
            "loss": all_loss,
            "preds": all_preds,
            "truths": all_truths,
            "perts": all_perts,
        }


def run_gears(
    train_adata: Any,
    valid_adata: Any,
    test_adata: Any,
    is_synthetic: bool,
    dataset_name: str = "dataset",
    model_dir: str = ".",
    perturbation_column: str = "perturbation",
    control_label: str = "control",
    expression_layer: str | None = None,
    device: str | None = None,
    use_gene_ontology_graph: bool = False,
    hidden_size: int = 64,
    num_go_gnn_layers: int = 1,
    num_gene_gnn_layers: int = 1,
    decoder_hidden_size: int = 16,
    uncertainty: bool = True,
    uncertainty_reg: float = 1.0,
    direction_lambda: float = 1.0,
    no_perturb: bool = False,
    G_coexpress: torch.Tensor | None = None,
    G_coexpress_weight: torch.Tensor | None = None,
    coexpress_threshold: float = 0.4,
    num_similar_genes_co_express_graph: int = 20,
    save_interval: int = 5,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    normalized_target_sum: float = 1e4,
    verbose: bool = False,
) -> dict[str, Any]:
    """Convenience method to run the full training and evaluation pipeline."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    train_adata = materialize_adata(train_adata, expression_layer=expression_layer)
    valid_adata = materialize_adata(valid_adata, expression_layer=expression_layer)
    test_adata = materialize_adata(test_adata, expression_layer=expression_layer)

    if verbose:
        print("train_adata shape:", train_adata.shape)
        print("train_adata obs", train_adata.obs.keys())
        print("train_adata layers", train_adata.layers.keys())
        print("train_adata obsm", train_adata.obsm.keys())
        print("train_adata var", train_adata.var.keys())
        if "cell_type" in train_adata.obs:
            print("Cell Types:", train_adata.obs["cell_type"].unique())
        print(f"Perturbations: {train_adata.obs[perturbation_column].nunique()}")
        print(
            f"Perturbation counts:\n{train_adata.obs[perturbation_column].value_counts().head(10)}"
        )

    # --- Train model ---
    # Add non_zeros_gene_idx to adata.uns for loss masking in GEARS
    # 1) Compute the mean expression across all cells for that perturbation condition.
    # 2) Find the indices of genes with non-zero mean expression.
    # 3) Store them sorted.

    train_adata.uns["non_zeros_gene_idx"] = get_non_zeros_gene_idx(
        adata=train_adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
    )

    node_map = {g: i for i, g in enumerate(train_adata.var_names)}

    # Map every gene to its graph node index so single-gene perturbations and
    # combo labels split into known gene tokens both resolve cleanly.
    node_map_pert = dict(node_map)
    control_label_pert_idx = len(
        node_map
    )  # assign control perturbation to a separate index after all genes
    node_map_pert[control_label] = control_label_pert_idx

    control_mask = train_adata.obs[perturbation_column].to_numpy(copy=False) == control_label
    if not np.any(control_mask):
        raise ValueError("GEARS requires at least one control cell in the training split.")
    control_expression_pool = train_adata.X[control_mask]
    if issparse(control_expression_pool):
        control_expression_pool = control_expression_pool.toarray()
    else:
        control_expression_pool = np.asarray(control_expression_pool)
    control_expression_pool = np.asarray(control_expression_pool, dtype=np.float32)
    train_batch_size = 64
    drop_last_train = check_last_singleton_batch(
        n_obs=int(train_adata.n_obs),
        batch_size=train_batch_size,
        model_name="GEARS",
    )
    if drop_last_train:
        warnings.warn(
            "Discarding the final GEARS training minibatch because it would contain a single cell "
            f"(n_train={train_adata.n_obs}, batch_size={train_batch_size}).",
            UserWarning,
            stacklevel=2,
        )

    train_loader = create_dataloader(
        adata=train_adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        control_label_pert_idx=control_label_pert_idx,
        node_map_pert=node_map_pert,
        control_expression_pool=control_expression_pool,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=drop_last_train,
    )

    valid_loader = None
    if valid_adata.n_obs > 0:
        valid_loader = create_dataloader(
            adata=valid_adata,
            perturbation_column=perturbation_column,
            control_label=control_label,
            control_label_pert_idx=control_label_pert_idx,
            node_map_pert=node_map_pert,
            control_expression_pool=control_expression_pool,
            batch_size=64,
            shuffle=False,
            drop_last=False,
        )

    test_loader = create_dataloader(
        adata=test_adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        control_label_pert_idx=control_label_pert_idx,
        node_map_pert=node_map_pert,
        control_expression_pool=control_expression_pool,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )

    # SET the perturbation graph to None to use co-expression graph instead of gene ontology graph since we don't have real perturbation targets in synthetic data
    if is_synthetic or not use_gene_ontology_graph:
        edge_index = None
        edge_weight = None
    else:
        gene_list = list(train_adata.var_names)
        go_edge_list = get_go_auto(
            gene_list=gene_list,
            data_path=model_dir,
            data_name=dataset_name,
        )
        if len(go_edge_list) > 0:
            go_graph = GeneSimNetwork(go_edge_list, gene_list, node_map)
            edge_index = go_graph.edge_index
            edge_weight = go_graph.edge_weight
        else:
            print("No GO graph edges found, using co-expression graph instead")
            edge_index = None
            edge_weight = None

    trainer = GEARS(
        pert_data=type(
            "PertData",
            (),
            {
                "dataloader": {
                    "train_loader": train_loader,
                    "val_loader": valid_loader,
                    "test_loader": test_loader,
                },
                "adata": train_adata,
                "perturbation_column": perturbation_column,
                "control_label": control_label,
                "node_map": node_map,  # gene name to index mapping
                "node_map_pert": node_map_pert,  # perturbation name to index mapping
                "data_path": model_dir,
                "dataset_name": dataset_name,
                "split": "train",
                "seed": _seed,
                "train_gene_set_size": train_adata.n_vars,
                "set2conditions": {"train": train_adata.obs[perturbation_column].unique().tolist()},
                "subgroup": None,
                "gene_names": train_adata.var_names.to_series(),
                "pert_names": train_adata.obs[perturbation_column].unique(),
                "default_pert_graph": {
                    "edge_index": edge_index,
                    "edge_weight": edge_weight,
                },
            },
        )(),
        device=resolved_device,
        weight_bias_track=False,
    )

    trainer.initialize_model(
        hidden_size=hidden_size,
        num_go_gnn_layers=num_go_gnn_layers,
        num_gene_gnn_layers=num_gene_gnn_layers,
        decoder_hidden_size=decoder_hidden_size,
        uncertainty=uncertainty,
        uncertainty_reg=uncertainty_reg,
        direction_lambda=direction_lambda,
        no_perturb=no_perturb,
        G_coexpress=G_coexpress,
        G_coexpress_weight=G_coexpress_weight,
        coexpress_threshold=coexpress_threshold,
        num_similar_genes_co_express_graph=num_similar_genes_co_express_graph,
        save_interval=save_interval,
    )

    print("Model:", trainer.model)
    trainer.train(epochs=epochs, lr=lr, weight_decay=weight_decay)

    if trainer.best_model is None:
        raise RuntimeError("best_model is None. Ensure training completed successfully.")
    out = trainer.predict(
        dataloader=test_loader,
        direction_lambda=direction_lambda,
        best_model=trainer.best_model,
    )
    # GEARS predicts on the normalized/log1p scale but does not constrain its
    # decoder outputs to remain a valid normalized expression profile. Project
    # them back to that manifold before downstream code treats this tensor as a
    # real `normalized_log1p` layer for DE and other metrics.
    preds_np: np.ndarray = out["preds"].detach().cpu().numpy()
    projected: np.ndarray = project_to_normalized_log1p(preds_np, target_sum=normalized_target_sum)

    control_mask = np.array(out["perts"]) == control_label
    if np.any(control_mask):
        projected[control_mask] = out["truths"].detach().cpu().numpy()[control_mask]

    out["preds"] = torch.from_numpy(projected)  # type: ignore[reportUnknownMemberType]
    return out


if __name__ == "__main__":
    # create synthetic data
    is_synthetic = True
    adata = anndata.AnnData()
    control_label = "control"
    perturbation_column = "perturbation"
    model_dir = "/workspaces/immunorep/immunorep-scrnaseq/data/gears/"
    # mapping requires perturbation names to exist in adata.var_names
    if is_synthetic:
        dataset_name = "synthetic"
        np.random.seed(_seed)  # noqa
        n_cells = 500
        n_genes = 200
        n_perturbations = 5

        # Gene names that will also serve as perturbation targets
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        # Pick first n_perturbations genes as perturbation targets
        pert_target_genes = gene_names[:n_perturbations]

        # Simulate count data
        counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))  # noqa

        # Assign each cell a perturbation (a target gene name) or control
        perturbations = np.random.choice(  # noqa
            [*pert_target_genes, control_label],
            size=n_cells,
        )

        # Inject perturbation signal: upregulate the target gene in perturbed cells
        for i, pert in enumerate(perturbations):
            if pert != control_label:
                target_idx = gene_names.index(pert)
                counts[i, target_idx] += np.random.poisson(50)  # noqa  # boost target gene

        adata = anndata.AnnData(
            X=csr_matrix(counts.astype(np.float32)),
            obs={perturbation_column: perturbations},
        )
        adata.var_names = gene_names
        adata.layers["counts"] = adata.X.copy()  # type: ignore
    else:
        # Load Norman19 data
        dataset_name = "norman19"
        adata = sc.read_h5ad(
            "/workspaces/immunorep/immunorep-scrnaseq/data/norman19/norman19_processed.h5ad"
        )
        print("Original Norman adata shape:", adata.shape)
        indices = np.random.choice(adata.n_obs, size=1000, replace=False)  # noqa
        adata = adata[indices].copy()

    # GEAR expects log-normalized data in adata.X see: https://github.com/snap-stanford/GEARS/blob/master/demo/data_tutorial.ipynb

if __name__ == "__main__":
    # create synthetic data
    is_synthetic = True
    adata = anndata.AnnData()
    control_label = "control"
    perturbation_column = "perturbation"
    model_dir = "/workspaces/immunorep/immunorep-scrnaseq/data/gears/"
    # mapping requires perturbation names to exist in adata.var_names
    if is_synthetic:
        dataset_name = "synthetic"
        adata = generate_synthetic_perturbation_data(
            perturbation_column=perturbation_column,
            control_label=control_label,
            context_key="cell_type",
        )
        adata.layers["counts"] = adata.X.copy()  # type: ignore
    else:
        # Load Norman19 data
        dataset_name = "norman19"
        adata = sc.read_h5ad(
            "/workspaces/immunorep/immunorep-scrnaseq/data/norman19/norman19_processed.h5ad"
        )
        print("Original Norman adata shape:", adata.shape)
        indices = np.random.choice(adata.n_obs, size=1000, replace=False)  # noqa
        adata = adata[indices].copy()

    if issparse(adata.X):  # type: ignore
        # For sparse matrices, only check non-zero entries
        data = adata.X.data  #  # type: ignore
        is_all_integer = np.allclose(data, np.round(data))  # type: ignore
    else:
        is_all_integer = np.allclose(adata.X, np.round(adata.X))  # type: ignore
    if is_all_integer:
        print("Data appears to be raw counts, applying normalization and log transformation...")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    else:
        print(
            "Data appears to be already normalized/log-transformed, proceeding without modification..."
        )
    # Split adata into train (60%), valid (20%), test (20%)
    if is_synthetic:
        n = adata.n_obs
        indices = np.random.permutation(n)  # noqa
        train_end = int(0.6 * n)
        valid_end = int(0.8 * n)
        train_adata = adata[indices[:train_end]].copy()
        valid_adata = adata[indices[train_end:valid_end]].copy()
        test_adata = adata[indices[valid_end:]].copy()
    else:
        # need to split by perturbation condition to ensure all conditions are represented in train/val/test for real data (this is within-context)
        perturbations = adata.obs[perturbation_column].unique()
        train_adata_list = []
        valid_adata_list = []
        test_adata_list = []
        for pert in perturbations:
            pert_adata = adata[adata.obs[perturbation_column] == pert].copy()
            n = pert_adata.n_obs
            if n < 3:
                # If too few cells for this perturbation, put all in train
                train_adata_list.append(pert_adata)  # type: ignore
                continue
            indices = np.random.permutation(n)  # noqa
            train_end = int(0.6 * n)
            valid_end = int(0.8 * n)
            train_adata_list.append(pert_adata[indices[:train_end]])  # type: ignore
            valid_adata_list.append(pert_adata[indices[train_end:valid_end]])  # type: ignore
            test_adata_list.append(pert_adata[indices[valid_end:]])  # type: ignore
        train_adata = anndata.concat(train_adata_list)  # type: ignore
        valid_adata = anndata.concat(valid_adata_list)  # type: ignore
        test_adata = anndata.concat(test_adata_list)  # type: ignore
    print(f"Split: train={train_adata.n_obs}, valid={valid_adata.n_obs}, test={test_adata.n_obs}")
    epochs = 50

    out = run_gears(
        train_adata=train_adata,
        valid_adata=valid_adata,
        test_adata=test_adata,
        is_synthetic=is_synthetic,
        hidden_size=64,
        num_go_gnn_layers=1,
        num_gene_gnn_layers=1,
        decoder_hidden_size=16,
        uncertainty=True,
        uncertainty_reg=1.0,
        direction_lambda=1.0,
        no_perturb=False,
        G_coexpress=None,  # will be constructed inside run() from adata
        G_coexpress_weight=None,  # will be constructed inside run() from adata
        coexpress_threshold=0.4,
        num_similar_genes_co_express_graph=20,
        save_interval=5,
        epochs=epochs,
        lr=1e-3,
        weight_decay=5e-4,
    )
    print(
        f"Test MSE: {out['loss']:.4f}, shape of predictions: {out['preds'].shape}, shape of truths: {out['truths'].shape} "
    )
    reconstructed = out["preds"].cpu().numpy()
    original = out["truths"].cpu().numpy()

    # Evaluate only on perturbed cells (controls are near-identity predictions)
    is_perturbed = np.asarray(test_adata.obs[perturbation_column]) != control_label
    print(
        f"Evaluating on {is_perturbed.sum()} perturbed cells out of {len(is_perturbed)} total test cells"
    )
    recon_mean = reconstructed[is_perturbed].mean(axis=0)
    orig_mean = original[is_perturbed].mean(axis=0)

    r, _ = pearsonr(
        orig_mean.A1 if hasattr(orig_mean, "A1") else orig_mean,
        recon_mean.A1 if hasattr(recon_mean, "A1") else recon_mean,
    )
    print(f"Pearson correlation between original and reconstructed mean expression: {r:.3f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    plt.scatter(orig_mean, recon_mean, alpha=0.3, s=5)  # type: ignore
    plt.xlabel("Original mean expression")  # type: ignore
    plt.ylabel("Reconstructed mean expression")  # type: ignore
    plt.title(f"Per-gene mean expression (Pearson r={r:.3f})")  # type: ignore
    plt.plot([0, orig_mean.max()], [0, orig_mean.max()], "r--")  # type: ignore
    # Diagonal line matching axis limits)
    plt.savefig(model_dir + "GEARS_gene_correlation.png")  # type: ignore
    # plt.show()
    print("Done!")
