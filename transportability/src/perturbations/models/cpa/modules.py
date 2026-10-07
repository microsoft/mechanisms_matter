"""Data module and related utilities for perturbation prediction models."""

import gc
import warnings

import lightning as L
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse  # type: ignore
from torch.utils.data import DataLoader

from .datasets import (
    SingleCellPerturbation,
    SingleCellPerturbationWithControls,
)
from .transform_ops import SingleCellPipeline
from .utils import batch_dataloader, noop_collate


class AnnDataLitModule(L.LightningDataModule):
    """AnnData Data Module for Perturbation Prediction Models."""

    def __init__(
        self,
        train_adata: sc.AnnData,
        val_adata: sc.AnnData | None,
        test_adata: sc.AnnData | None,
        add_controls: bool,
        perturbation_key: str,
        perturbation_control_value: str,
        batch_size: int,
        perturbation_combination_delimiter: str | None,
        covariate_keys: list[str] | None = None,
        num_workers: int = 0,
        num_val_workers: int | None = None,
        num_test_workers: int | None = None,
        use_counts: bool = False,
        embedding_key: str | None = None,
    ) -> None:
        """Initialize the AnnDataLitModule."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_workers = num_val_workers if num_val_workers is not None else num_workers
        self.num_test_workers = num_test_workers if num_test_workers is not None else num_workers
        self.split = None  ## Data split as a pandas series

        def _from_anndata(adata: sc.AnnData):
            if add_controls:
                return SingleCellPerturbationWithControls.from_anndata(
                    adata,
                    perturbation_key=perturbation_key,
                    perturbation_combination_delimiter=perturbation_combination_delimiter,
                    covariate_keys=covariate_keys,
                    perturbation_control_value=perturbation_control_value,
                    embedding_key=embedding_key,
                )
            else:
                return SingleCellPerturbation.from_anndata(
                    adata,
                    perturbation_key=perturbation_key,
                    perturbation_combination_delimiter=perturbation_combination_delimiter,
                    perturbation_control_value=perturbation_control_value,
                    covariate_keys=covariate_keys,
                    embedding_key=embedding_key,
                )

        # assert that the adata objects have raw counts in .X
        for ad in [train_adata, val_adata, test_adata]:
            if ad is None:
                continue
            if issparse(ad.X):  # type: ignore
                # For sparse matrices, only check non-zero entries
                data = ad.X.data  #  # type: ignore
                is_all_integer = np.allclose(data, np.round(data))  # type: ignore
            else:
                is_all_integer = np.allclose(ad.X, np.round(ad.X))  # type: ignore
            if use_counts:
                assert is_all_integer, (
                    "adata.X contains non-integer values — expected raw counts. "
                    "If your data is log-normalized, set use_counts=False. "
                    "If your data is raw counts but contains some non-integer values, "
                    "please round the counts to integers before creating the dataset."
                )
            else:
                assert not is_all_integer, (
                    "adata.X contains only integer values — this looks like raw counts. "
                    "Expected float values after log1p normalization. "
                    "Run sc.pp.normalize_total(adata) and sc.pp.log1p(adata) first, or set use_counts=True if you want to use raw counts."
                )

        # Create datasets
        self.train_dataset, train_context = _from_anndata(train_adata)

        self.val_dataset, val_context = (
            _from_anndata(val_adata) if val_adata is not None else (None, None)
        )

        self.test_dataset, test_context = (
            _from_anndata(test_adata) if test_adata is not None else (None, None)
        )
        self.train_context = train_context

        # Verify that train, val, test datasets have the same perturbations and covariates
        self._verify_splits(train_context, val_context, test_context)  # type: ignore

        self.example_collate_fn = noop_collate()

        self.num_perturbations = len(train_context["perturbation_uniques"])
        # After creating datasets and train_context:
        transform = SingleCellPipeline(
            perturbation_uniques=set(train_context["perturbation_uniques"]),
            covariate_uniques={k: set(v) for k, v in train_context["covariate_uniques"].items()},
        )

        self.train_dataset.transform = transform
        # print(f"Train Transform set: {self.train_dataset.transform}")
        # print(f"Train Transform type: {type(self.train_dataset.transform)}")
        if self.val_dataset is not None:
            self.val_dataset.transform = transform
            # print(f"Val Transform set: {self.val_dataset.transform}")
            # print(f"Val Transform type: {type(self.val_dataset.transform)}")
        if self.test_dataset is not None:
            self.test_dataset.transform = transform
            # print(f"Test Transform set: {self.test_dataset.transform}")
            # print(f"Test Transform type: {type(self.test_dataset.transform)}")

        # Cleanup
        del train_adata
        if val_adata is not None:
            del val_adata
        if test_adata is not None:
            del test_adata
        gc.collect()

    @property
    def num_covariates(self) -> int:
        """Number of unique covariate keys."""
        return len(self.train_context.get("covariate_uniques", {}))

    @property
    def num_genes(self) -> int:
        """Number of genes in the dataset."""
        if self.train_dataset.gene_names is None:
            raise ValueError("gene_names is not set on the training dataset.")
        return len(self.train_dataset.gene_names)

    @property
    def embedding_width(self) -> int | None:
        """Width of the embeddings, or None if not set."""
        if self.train_dataset.embeddings is None:
            return None
        return self.train_dataset.embeddings.shape[1]

    @staticmethod
    def _verify_splits(train_info: dict, val_info: dict | None, test_info: dict | None):  # type: ignore
        for split, info in [("val", val_info), ("test", test_info)]:  # type: ignore
            if info is not None:
                if not set(train_info["perturbation_uniques"]) >= set(info["perturbation_uniques"]):  # type: ignore
                    raise RuntimeError(
                        f"Train dataset must contain all perturbations in {split} dataset."
                    )
                if set(train_info["perturbation_uniques"]) != set(info["perturbation_uniques"]):  # type: ignore
                    warnings.warn(
                        f"{split} dataset is missing perturbations from train dataset.",
                        stacklevel=2,
                    )

                if not set(train_info["covariate_uniques"]) >= set(info["covariate_uniques"]):  # type: ignore
                    raise RuntimeError(
                        f"Train dataset must contain all covariates in {split} dataset."
                    )
                if set(train_info["covariate_uniques"]) != set(info["covariate_uniques"]):  # type: ignore
                    warnings.warn(
                        f"{split} dataset is missing covariates from train dataset.", stacklevel=2
                    )

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        """Return the training dataloader."""
        return batch_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.example_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[torch.Tensor] | None:
        """Return the validation dataloader, or None if no validation dataset is provided."""
        if self.val_dataset is None:
            return None
        else:
            return batch_dataloader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_val_workers,
                collate_fn=self.example_collate_fn,
            )

    def test_dataloader(self) -> DataLoader[torch.Tensor] | None:
        """Return the test dataloader, or None if no test dataset is provided."""
        if self.test_dataset is None:
            return None
        else:
            return batch_dataloader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_test_workers,
                shuffle=False,
                collate_fn=self.example_collate_fn,
            )
