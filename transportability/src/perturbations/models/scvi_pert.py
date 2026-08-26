"""scVI perturbation models and utilities for single-cell analysis."""

import gc
import warnings
from typing import Literal, cast

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy.stats import pearsonr  # type: ignore
from scvi.dataloaders import CollectionAdapter

from .test_synthentic_data import generate_synthetic_perturbation_data

torch.set_float32_matmul_precision("medium")

_seed = 42
torch.manual_seed(_seed)  # type: ignore
np.random.seed(_seed)  # noqa
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)


class ScviPerturbation:
    """Train scVI on train data and infer on test data."""

    def __init__(
        self,
        data: anndata.AnnData | anndata.experimental.AnnCollection,
        counts_layer: str | None = None,
        control_label: str = "control",
        perturbation_key: str = "perturbation",
        context_key: str = "cell_line",
        batch_key: str | None = None,
        train_idx: np.ndarray | None = None,
        val_idx: np.ndarray | None = None,
        test_idx: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize a scVI perturbation runner with optional external split indices."""
        self.data: anndata.AnnData | CollectionAdapter = self._adapt_data(data)
        self.counts_layer = counts_layer
        self.control_label = control_label
        self.perturbation_key = perturbation_key
        self.context_key = context_key
        self.batch_key = batch_key
        self.seed = seed

        self.model: scvi.model.SCVI | None = None
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        self._compact_external_indexing()

    @staticmethod
    def _adapt_data(
        data: anndata.AnnData | anndata.experimental.AnnCollection | CollectionAdapter,
    ) -> anndata.AnnData | CollectionAdapter:
        if isinstance(data, (anndata.AnnData, CollectionAdapter)):
            return data
        else:
            return CollectionAdapter(data)

    @staticmethod
    def _normalize_indices(indices: np.ndarray | None) -> np.ndarray | None:
        if indices is None:
            return None
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            idx = idx.ravel()
        return idx

    def _compact_external_indexing(self) -> None:
        self.train_idx = self._normalize_indices(self.train_idx)
        self.val_idx = self._normalize_indices(self.val_idx)
        self.test_idx = self._normalize_indices(self.test_idx)

        parts = [idx for idx in (self.train_idx, self.val_idx, self.test_idx) if idx is not None]
        if not parts:
            return

        n_obs = int(self.data.n_obs)
        used_idx = np.unique(np.concatenate(parts, dtype=np.int64))
        if used_idx.size == n_obs and np.array_equal(
            used_idx,
            np.arange(n_obs, dtype=np.int64),
        ):
            return

        remap: np.ndarray = np.full(n_obs, -1, dtype=np.int64)  # type: ignore[reportUnknownMemberType]
        remap[used_idx] = np.arange(used_idx.size, dtype=np.int64)

        if isinstance(self.data, CollectionAdapter):
            subset_view = self.data.collection[used_idx, :]  # type: ignore[reportUnknownMemberType]
            # scvi.setup_anndata writes bookkeeping columns into obs, which an
            # AnnCollectionView does not allow. Materialize only the compacted
            # subset so external indexing can remain gap-free and mutable.
            self.data = cast(anndata.AnnData, subset_view.to_adata())  # type: ignore[reportUnknownMemberType]
        else:
            self.data = self.data[used_idx, :].copy()

        def _remap(indices: np.ndarray | None) -> np.ndarray | None:
            if indices is None:
                return None
            mapped = remap[indices]
            if np.any(mapped < 0):
                raise ValueError("Encountered unmapped scVI external indices after compaction.")
            return mapped

        self.train_idx = _remap(self.train_idx)
        self.val_idx = _remap(self.val_idx)
        self.test_idx = _remap(self.test_idx)

    def run(
        self,
        normalized_target_sum: float = 1e4,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        max_epochs: int = 50,
        batch_size: int = 256,
        early_stopping: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_persistent_workers: bool | None = None,
    ) -> anndata.AnnData:
        """Train scVI and return test cells with latent and normalized embeddings."""
        scvi.model.SCVI.setup_anndata(  # type: ignore[reportUnknownMemberType]  # type: ignore[reportUnknownMemberType]
            self.data,  # type: ignore[arg-type]  # type: ignore[arg-type]
            layer=self.counts_layer,
            batch_key=self.batch_key,
            categorical_covariate_keys=[self.perturbation_key, self.context_key],
        )
        n_train = int(self.train_idx.size if self.train_idx is not None else self.data.n_obs)
        if n_train == 1:
            raise ValueError(
                "scVI received only 1 training cell. Dropping a singleton residual minibatch would remove the entire training set."
            )
        drop_last = n_train > batch_size and (n_train % batch_size == 1)
        if drop_last:
            warnings.warn(
                f"Discarding the final scVI training minibatch because it would contain a single cell (n_train={n_train}, batch_size={batch_size}).",
                UserWarning,
                stacklevel=2,
            )
        self.model = scvi.model.SCVI(
            self.data,  # type: ignore[arg-type]
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            dispersion=dispersion,
        )
        if dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be >= 0.")
        if dataloader_persistent_workers is None:
            dataloader_persistent_workers = dataloader_num_workers > 0

        datasplitter_kwargs: dict[str, object] = {
            "external_indexing": [self.train_idx, self.val_idx, self.test_idx],
            # Keep this at 0 by default to avoid nested multiprocessing semaphore
            # issues when the outer sweep already runs in many worker processes.
            "num_workers": int(dataloader_num_workers),
            "drop_last": bool(drop_last),
        }
        if dataloader_num_workers > 0:
            datasplitter_kwargs["persistent_workers"] = bool(dataloader_persistent_workers)

        self.model.train(  # type: ignore[reportUnknownMemberType]
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            datasplitter_kwargs=datasplitter_kwargs,
        )
        assert self.model is not None
        assert self.train_idx is not None
        assert self.test_idx is not None
        z_train = np.asarray(
            self.model.get_latent_representation(indices=self.train_idx),  # type: ignore[reportUnknownMemberType]
            dtype=np.float32,
        )

        if isinstance(self.data, anndata.AnnData):
            obs = cast("pd.DataFrame", self.data.obs)
        else:
            obs = cast("pd.DataFrame", self.data.collection.obs)  # type: ignore[reportUnknownMemberType]
        train_obs: pd.DataFrame = obs.iloc[self.train_idx]  # type: ignore[assignment]
        test_obs: pd.DataFrame = obs.iloc[self.test_idx]  # type: ignore[assignment]
        train_perts: np.ndarray = np.asarray(train_obs[self.perturbation_key].values)  # type: ignore[reportUnknownArgumentType]
        test_perts: np.ndarray = np.asarray(test_obs[self.perturbation_key].values)  # type: ignore[reportUnknownArgumentType]

        # Identify control cells (perturbation == "control")
        control_mask_train = train_perts == self.control_label
        if not np.any(control_mask_train):
            raise ValueError(
                f"No control cells found in training set (looked for {self.control_label} in {self.perturbation_key})."
            )

        # Compute mean control z per context (cell type), or global if no context_key
        if self.context_key in train_obs.columns:  # type: ignore[reportUnknownMemberType]
            train_contexts: np.ndarray | None = np.asarray(train_obs[self.context_key].values)  # type: ignore[reportUnknownArgumentType]
            test_contexts: np.ndarray | None = np.asarray(test_obs[self.context_key].values)  # type: ignore[reportUnknownArgumentType]
            unique_contexts = np.unique(train_contexts[control_mask_train])
            z_control_by_ctx = {
                ctx: z_train[control_mask_train & (train_contexts == ctx)].mean(axis=0)
                for ctx in unique_contexts
            }
            # Fallback for unseen contexts
            z_control_global = z_train[control_mask_train].mean(axis=0)
        else:
            train_contexts = None
            test_contexts = None
            z_control_global = z_train[control_mask_train].mean(axis=0)
            z_control_by_ctx = {}

        # Compute delta_z per perturbation per context
        unique_perts = np.unique(train_perts[~control_mask_train])
        delta_z: dict[tuple[str, str], np.ndarray] = {}
        for p in unique_perts:
            p_mask = train_perts == p
            if train_contexts is not None:
                for ctx in np.unique(train_contexts[p_mask]):
                    ctx_ctrl = z_control_by_ctx.get(ctx, z_control_global)
                    ctx_pert = z_train[p_mask & (train_contexts == ctx)].mean(axis=0)
                    delta_z[(p, ctx)] = ctx_pert - ctx_ctrl
            else:
                z_pert_mean = z_train[p_mask].mean(axis=0)
                delta_z[(p, "")] = z_pert_mean - z_control_global

        # Get test latent representations
        z_test = np.asarray(
            self.model.get_latent_representation(indices=self.test_idx),  # type: ignore[reportUnknownMemberType]
            dtype=np.float32,
        )

        # Predict: for each test cell, apply delta_z based on its perturbation and context
        z_predicted = np.zeros_like(z_test)
        for i, (pert, z_i) in enumerate(zip(test_perts, z_test, strict=True)):
            if pert == self.control_label:
                z_predicted[i] = z_i
            else:
                ctx = str(test_contexts[i]) if test_contexts is not None else ""
                key = (pert, ctx)
                if key in delta_z:
                    z_ctrl = (
                        z_control_by_ctx.get(ctx, z_control_global)
                        if test_contexts is not None
                        else z_control_global
                    )
                    z_predicted[i] = z_ctrl + delta_z[key]
                elif any(k[0] == pert for k in delta_z):
                    # Known perturbation but unseen context — average across available contexts
                    matching = [v for k, v in delta_z.items() if k[0] == pert]
                    avg_delta = np.mean(matching, axis=0)
                    z_ctrl = (
                        z_control_by_ctx.get(ctx, z_control_global)
                        if test_contexts is not None
                        else z_control_global
                    )
                    z_predicted[i] = z_ctrl + avg_delta
                else:
                    # Unseen perturbation — keep observed z
                    z_predicted[i] = z_i

        # Decode z_predicted back to gene expression via the scVI generative module
        device = next(self.model.module.parameters()).device  # type: ignore[reportUnknownMemberType]
        z_predicted_tensor = torch.tensor(z_predicted, dtype=torch.float32).to(device)
        # Use normalized_target_sum as library size (log-space), matching get_normalized_expression
        library = torch.full(
            (z_predicted.shape[0], 1),
            np.log(normalized_target_sum),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            batch_index = torch.zeros(z_predicted.shape[0], 1, dtype=torch.long, device=device)
            # Read cat_covs from scVI's registered encoding in obsm
            scvi_cat_covs: np.ndarray = (  # type: ignore[reportUnknownMemberType]
                self.data.obsm["_scvi_extra_categorical_covs"]
                if isinstance(self.data, anndata.AnnData)
                else self.data.collection.obsm["_scvi_extra_categorical_covs"]  # type: ignore[reportUnknownMemberType]
            )
            cat_covs_np: np.ndarray = np.asarray(scvi_cat_covs)[self.test_idx]
            cat_covs = torch.tensor(cat_covs_np, dtype=torch.long, device=device)
            generative_outputs = self.model.module.generative(  # type: ignore[reportUnknownMemberType]
                z=z_predicted_tensor,
                library=library,
                batch_index=batch_index,
                cat_covs=cat_covs,
            )
        predicted_expression: np.ndarray = np.asarray(
            generative_outputs["px"].mean.cpu().numpy()  # type: ignore[reportUnknownMemberType]
        )
        normalized_log1p = np.log1p(predicted_expression).astype(np.float32)

        if isinstance(self.data, CollectionAdapter):
            # IMPORTANT: AnnCollectionView.to_adata() gives X/layers; AnnCollection.to_adata() does not.
            test_view = self.data.collection[self.test_idx, :]  # type: ignore[reportUnknownMemberType]
            adata_out: anndata.AnnData = cast(anndata.AnnData, test_view.to_adata())  # type: ignore[reportUnknownMemberType]
        else:
            # regular AnnData case
            adata_out = self.data[self.test_idx].copy()

        adata_out.obsm["X_scvi"] = z_predicted
        adata_out.layers["normalized_log1p"] = normalized_log1p

        # Replace decoded predictions for control cells with ground truth expression
        control_mask_test = np.asarray(adata_out.obs[self.perturbation_key]) == self.control_label
        if np.any(control_mask_test):
            ctrl_view = adata_out[control_mask_test]
            ctrl_counts = (
                ctrl_view.X if self.counts_layer is None else ctrl_view.layers[self.counts_layer]
            )
            ctrl_dense = np.asarray(
                ctrl_counts.todense() if hasattr(ctrl_counts, "todense") else ctrl_counts,  # type: ignore[reportUnknownMemberType]
                dtype=np.float32,
            )
            lib_sizes = ctrl_dense.sum(axis=1, keepdims=True)
            lib_sizes[lib_sizes == 0] = 1.0
            adata_out.layers["normalized_log1p"][control_mask_test] = np.log1p(
                ctrl_dense / lib_sizes * normalized_target_sum
            ).astype(np.float32)

        return adata_out

    def close(self) -> None:
        """Release trained model, trainer, registry, and accelerator references."""
        model = self.model
        self.model = None
        if model is None:
            return

        model.to_device("cpu")
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            trainer._model = None
            model.trainer = None
        if model.adata is not None:
            model.deregister_manager(model.adata)

        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # create synthetic data
    is_synthetic = True
    adata = anndata.AnnData()
    perturbation_column = "perturbation"
    context_key = "cell_type"
    counts_layer = "counts"
    control_label = "control"
    normalized_target_sum = 1e4
    epochs = 100
    if is_synthetic:
        adata = generate_synthetic_perturbation_data(
            perturbation_column=perturbation_column,
            control_label=control_label,
            context_key=context_key,
        )
        adata.layers[counts_layer] = adata.X.copy()  # type: ignore
    else:
        # Load Norman19 data
        adata = sc.read_h5ad(
            "/workspaces/immunorep/immunorep-scrnaseq/data/norman19/norman19_processed.h5ad"
        )
        print("Original Norman adata shape:", adata.shape)
        indices = np.random.choice(adata.n_obs, size=1000, replace=False)  # noqa
        print(f"Subsetting to {len(indices)} random cells for testing...")
        adata = adata[indices].copy()
        adata.obs[context_key] = np.array(["K562"] * adata.n_obs)

    print("adata shape:", adata.shape)
    print("adata obs", adata.obs.keys())
    print("adata layers", adata.layers.keys())
    print("adata obsm", adata.obsm.keys())
    print("adata var", adata.var.keys())
    print("Cell Types:", adata.obs["cell_type"].unique())
    print(f"Perturbations: {adata.obs['perturbation'].nunique()}")
    print(f"Perturbation counts:\n{adata.obs['perturbation'].value_counts().head(10)}")

    # Split adata into train (60%), valid (20%), test (20%)

    n = adata.n_obs
    indices = np.random.permutation(n)  # noqa
    train_end = int(0.6 * n)
    valid_end = int(0.8 * n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:valid_end]
    test_idx = indices[valid_end:]

    # --- Train model ---
    model = ScviPerturbation(
        adata,
        control_label=control_label,
        counts_layer=counts_layer,
        perturbation_key=perturbation_column,
        context_key=context_key,
        batch_key=None,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=_seed,
    )
    result = model.run(
        n_latent=10,
        n_hidden=128,
        n_layers=1,
        gene_likelihood="zinb",
        max_epochs=epochs,
        batch_size=256,
        early_stopping=True,
        normalized_target_sum=normalized_target_sum,
    )
    print(model.model.summary_string)  # type: ignore
    print(model.model.history.keys())  # type: ignore
    print(model.model.history["train_loss"].tail(10))  # type: ignore

    #    # result is the test-set AnnData with X_scvi and normalized_log1p
    print(f"Test result shape: {result.shape}")
    print(f"Latent shape: {result.obsm['X_scvi'].shape}")
    print(f"Normalized log1p shape: {result.layers['normalized_log1p'].shape}")

    # Check scales match
    print("Counts range:", result.layers["counts"].min(), result.layers["counts"].max())
    print(
        "Normalized log1p range:",
        result.layers["normalized_log1p"].min(),
        result.layers["normalized_log1p"].max(),
    )

    # Mean expression per gene: original counts vs normalized_log1p
    original_counts = result.layers["counts"]
    # Normalize counts to target_sum + log1p to match predicted scale
    counts_dense = np.asarray(
        original_counts.todense() if hasattr(original_counts, "todense") else original_counts,  # type: ignore
        dtype=np.float32,
    )
    lib_sizes = counts_dense.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0  # avoid division by zero
    original_normalized = np.log1p(counts_dense / lib_sizes * normalized_target_sum)

    reconstructed = np.array(result.layers["normalized_log1p"])
    is_control = np.asarray(result.obs[perturbation_column]) == control_label
    is_perturbed = ~is_control
    print(f"Number of control cells: {is_control.sum()}")
    print(f"Number of perturbed cells: {is_perturbed.sum()}")

    # Evaluate only on perturbed cells (controls are identity, not predictions)
    orig_mean = original_normalized[is_perturbed].mean(axis=0)
    recon_mean = reconstructed[is_perturbed].mean(axis=0)

    r, _ = pearsonr(
        orig_mean.A1 if hasattr(orig_mean, "A1") else orig_mean.ravel(),
        recon_mean.A1 if hasattr(recon_mean, "A1") else recon_mean.ravel(),
    )
    print(f"Pearson correlation (counts vs normalized_log1p): {r:.3f}")

    plt.scatter(orig_mean, recon_mean, alpha=0.3, s=5)  # type: ignore
    plt.xlabel("Original normalized log1p mean")  # type: ignore
    plt.ylabel("Normalized log1p mean")  # type: ignore
    plt.title(f"Per-gene mean expression (Pearson r={r:.3f})")  # type: ignore
    model_dir = "/workspaces/immunorep/immunorep-scrnaseq/src/immunorep/perturbations/models/"
    plt.savefig(model_dir + "SCVI_gene_correlation.png")  # type: ignore
    print("Done!")
