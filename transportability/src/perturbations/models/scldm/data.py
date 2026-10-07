"""Data module for scLDM Perturb-seq asSTATE datasets using in-memory AnnData objects."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pytorch_lightning import LightningDataModule
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Dataset

from .constants import ModelEnum
from .encoder import VocabularyEncoderSimplified

CONDITION_KEYS = ("cell_line", "gene")


# pyright: reportUnknownMemberType=false
def _to_dense(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert a sparse matrix to a dense numpy array, or return the array as-is if already dense."""
    x_densed = x.todense() if sp.issparse(x) else x  # type: ignore
    return np.asarray(x_densed, dtype=np.float32)


def tokenize_cells(
    cell: np.ndarray,
    var_names: Sequence[str],
    encoder: VocabularyEncoderSimplified,
    genes_seq_len: int,
    sample_genes: Literal[
        "random", "weighted", "expressed", "expressed_zero", "random_expressed", "none"
    ],
    gene_tokens_key: str = ModelEnum.GENES.value,
    counts_key: str = ModelEnum.COUNTS.value,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Tokenize cell counts into gene tokens.

    Parameters
    ----------
    cell
        Count matrix of shape (N, G) where N is number of cells, G is number of genes
    var_names
        Gene names corresponding to columns of cell
    encoder
        Vocabulary encoder to map gene names to token indices
    genes_seq_len
        Maximum sequence length for sampled genes
    sample_genes
        Sampling strategy for genes
    gene_tokens_key
        Key for gene tokens in output dict
    counts_key
        Key for counts in output dict
    seed
        Random seed for reproducibility

    Returns:
    -------
    dict[str, np.ndarray]
        Dictionary with tokenized genes, counts, and library sizes
    """
    counts = cell
    gene_idx = np.tile(encoder.encode_genes(var_names), (len(counts), 1))
    library_size = counts.sum(1, keepdims=True)

    rng = np.random.default_rng(seed=seed)
    N, G = counts.shape

    if sample_genes == "weighted":
        if encoder.metadata_genes is None:
            raise ValueError("encoder.metadata_genes must be set for weighted sampling")

        metadata_genes = cast("pd.DataFrame", encoder.metadata_genes)
        scaled_counts = (counts + 1) / metadata_genes["means"].values
        scaled_counts = scaled_counts / scaled_counts.sum(1, keepdims=True)
        sampled_idx = np.stack(
            [rng.choice(G, size=genes_seq_len, replace=False, p=p) for p in scaled_counts]
        )
        return {
            gene_tokens_key: np.take_along_axis(gene_idx, sampled_idx, axis=1),
            counts_key: np.take_along_axis(counts, sampled_idx, axis=1),
            "library_size": library_size,
        }

    elif sample_genes == "expressed":
        mask_idx = encoder.mask_token_idx
        expressed = counts > 0
        num_expressed = expressed.sum(axis=1)

        if (num_expressed > genes_seq_len).any():
            raise ValueError("genes_seq_len is smaller than number of expressed genes")

        pos_order = expressed.cumsum(axis=1) - 1
        genes_out = np.full((N, genes_seq_len), mask_idx, dtype=gene_idx.dtype)
        counts_out = np.zeros((N, genes_seq_len), dtype=counts.dtype)

        ii, jj = np.where(expressed)
        pp = pos_order[expressed]
        genes_out[ii, pp] = gene_idx[ii, jj]
        counts_out[ii, pp] = counts[ii, jj]

        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            ModelEnum.GENES_SUBSET.value: genes_out,
            ModelEnum.COUNTS_SUBSET.value: counts_out,
            "library_size": library_size,
        }

    elif sample_genes == "expressed_zero":
        expressed = counts > 0
        permuted_indices = np.stack([rng.permutation(G) for _ in range(N)])

        shuffled_gene_idx = np.take_along_axis(gene_idx, permuted_indices, axis=1)
        shuffled_counts = np.take_along_axis(counts, permuted_indices, axis=1)
        shuffled_expressed = np.take_along_axis(expressed, permuted_indices, axis=1)

        priority = shuffled_expressed.astype(int)
        sort_indices = np.argsort(priority, axis=1, kind="stable")

        final_gene_idx = np.take_along_axis(shuffled_gene_idx, sort_indices, axis=1)
        final_counts = np.take_along_axis(shuffled_counts, sort_indices, axis=1)

        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            ModelEnum.GENES_SUBSET.value: final_gene_idx[:, :genes_seq_len],
            ModelEnum.COUNTS_SUBSET.value: final_counts[:, :genes_seq_len],
            "library_size": library_size,
        }

    elif sample_genes == "random_expressed":
        mask_idx = encoder.mask_token_idx
        nonzero_mask = counts > 0

        sampled_idx = np.stack(
            [
                np.pad(
                    rng.choice(
                        np.nonzero(nonzero_mask[i])[0],
                        size=min(genes_seq_len, nonzero_mask[i].sum()),
                        replace=False,
                    ),
                    (0, max(0, genes_seq_len - nonzero_mask[i].sum())),
                    constant_values=-1,
                )
                for i in range(N)
            ]
        )

        padded_mask = sampled_idx == -1
        safe_sampled_idx = np.where(padded_mask, 0, sampled_idx)

        sampled_gene_idx = np.take_along_axis(gene_idx, safe_sampled_idx, axis=1)
        subset_counts = np.take_along_axis(counts, safe_sampled_idx, axis=1)

        sampled_gene_idx[padded_mask] = mask_idx
        subset_counts[padded_mask] = 0

        return {
            gene_tokens_key: sampled_gene_idx,
            counts_key: subset_counts,
            "library_size": library_size,
        }

    elif sample_genes == "random":
        sampled_idx = np.stack([rng.choice(G, size=genes_seq_len, replace=False) for _ in range(N)])
        return {
            gene_tokens_key: np.take_along_axis(gene_idx, sampled_idx, axis=1),
            counts_key: np.take_along_axis(counts, sampled_idx, axis=1),
            "library_size": library_size,
        }

    elif sample_genes == "none":
        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            "library_size": library_size,
        }

    else:
        raise ValueError(f"Invalid sample_genes value: {sample_genes}")


class _PerturbseqDataset(Dataset[dict[str, np.ndarray]]):
    """
    Yields per-chunk batch dicts already tokenized (chunk = one dataloader item).

    We tokenize a contiguous chunk of ``chunk_size`` cells per ``__getitem__`` so the expensive
    ``tokenize_cells`` tiling runs on blocks (like the upstream collated batches), then a trivial
    ``collate`` concatenates chunks. ``batch_size`` at the DataLoader level = number of chunks;
    effective cell batch = ``batch_size * chunk_size``. For simplicity we set chunk_size = the
    desired cell batch and DataLoader batch_size = 1 (one chunk per step).
    """

    def __init__(
        self,
        adata: ad.AnnData,
        encoder: VocabularyEncoderSimplified,
        chunk_size: int,
        shuffle: bool,
        seed: int = 42,
    ):
        self.adata = adata
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.var_names = list(adata.var_names)
        self._labels = {k: adata.obs[k].to_numpy() for k in CONDITION_KEYS}
        n = adata.n_obs
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        self.chunks = [idx[i : i + chunk_size] for i in range(0, n, chunk_size)]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        rows = self.chunks[i]
        counts = _to_dense(self.adata.X[rows])  # type: ignore
        tok = tokenize_cells(
            cell=counts,
            var_names=self.var_names,
            encoder=self.encoder,
            genes_seq_len=self.adata.n_vars,
            sample_genes="none",
        )
        genes = tok[ModelEnum.GENES.value].astype(np.int64)
        counts_arr = tok[ModelEnum.COUNTS.value].astype(np.float32)
        out: dict[str, np.ndarray] = {
            ModelEnum.COUNTS.value: counts_arr,
            ModelEnum.GENES.value: genes,
            ModelEnum.LIBRARY_SIZE.value: tok["library_size"].astype(np.float32),
            # All genes are used ("all"): the VAE encoder's gene *subset* is the full
            # gene set. The model's forward feeds counts_subset/genes_subset to the encoder input
            # layer (vae.forward -> InputTransformerVAE), so for the no-subsampling case the subset
            # equals the full counts/genes. (The DiT ignores condition keys outside its
            # class_vocab_sizes, so carrying these in the LDM batch is harmless.)
            ModelEnum.COUNTS_SUBSET.value: counts_arr,
            ModelEnum.GENES_SUBSET.value: genes,
        }
        for k in CONDITION_KEYS:
            out[k] = self.encoder.encode_metadata(self._labels[k][rows], label=k).astype(np.int64)
        return out


def _as_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(a)


def _collate(chunks: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Concatenate pre-tokenized chunks along the cell axis and to-tensor (mirrors collate_fn)."""
    keys = chunks[0].keys()
    out = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
    return tree_map(_as_tensor, out)


class PerturbseqDataModule(LightningDataModule):
    """DataModule providing pre-tokenized Perturb-seq train, validation, and test loaders."""

    def __init__(
        self,
        train_h5ad: str | Path | None,
        test_h5ad: str | Path,
        metadata_json: str | Path,
        mu_size_factor: str | Path,
        sd_size_factor: str | Path,
        val_h5ad: str | Path | None = None,
        batch_size: int = 128,
        test_batch_size: int = 256,
        num_workers: int = 0,
        seed: int = 42,
    ):
        """Initialize the PerturbseqDataModule with paths to train/test data, metadata, size factors, and DataLoader parameters."""
        super().__init__()
        self.train_h5ad = str(train_h5ad) if train_h5ad is not None else None
        self.test_h5ad = str(test_h5ad)
        self.val_h5ad = str(val_h5ad) if val_h5ad is not None else None
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed
        # The vocabulary encoder is the authoritative gene/label vocab + size-factor source.
        # Built from the committed metadata JSON (genes + label categories) — joint conditioning.
        self.vocabulary_encoder = VocabularyEncoderSimplified(
            adata_path=None,
            # Only the label *keys* are consumed by the encoder (it derives the real category
            # lists from the metadata JSON); the values here are unused placeholders. The DiT's
            # actual embedding sizes come from the data-driven ``class_vocab_sizes`` property.
            class_vocab_sizes=dict.fromkeys(CONDITION_KEYS, 0),
            n_genes=None,
            condition_strategy="joint",
            mu_size_factor=str(mu_size_factor),
            sd_size_factor=str(sd_size_factor),
            metadata_json=str(metadata_json),
        )
        self._train: _PerturbseqDataset | None = None
        self._test: _PerturbseqDataset | None = None
        self._val: _PerturbseqDataset | None = None
        self.n_cells = 0

    @property
    def class_vocab_sizes(self) -> dict[str, int]:
        """
        Per-label DiT vocabulary sizes, derived from the metadata categories (data-driven).

        Each size is the number of distinct categories for that condition key; the DiT reserves one
        extra index (== size) as the classifier-free-guidance null token. Use this to size the DiT
        condition embeddings so the architecture always matches the dataset (no hardcoded values).
        """
        # The encoder is always built from ``metadata_json`` here, so ``labels`` is never None.
        labels = cast("dict[str, list[str]]", self.vocabulary_encoder.labels)
        return {k: len(labels[k]) for k in CONDITION_KEYS}

    @property
    def n_genes(self) -> int:
        """Number of gene tokens encoded by the asSTATE metadata for this dataset."""
        return int(cast("int", self.vocabulary_encoder.n_genes))

    def setup(self, stage: str | None = None):
        """Set up the datasets required by the requested Lightning stage."""
        if stage in (None, "fit") and self.train_h5ad is not None and self._train is None:
            adata = ad.read_h5ad(self.train_h5ad)
            self.n_cells = adata.n_obs
            self._train = _PerturbseqDataset(
                adata,
                self.vocabulary_encoder,
                chunk_size=self.batch_size,
                shuffle=True,
                seed=self.seed,
            )
        if stage in (None, "fit", "predict", "test") and self._test is None:
            adata = ad.read_h5ad(self.test_h5ad)
            self._test = _PerturbseqDataset(
                adata,
                self.vocabulary_encoder,
                chunk_size=self.test_batch_size,
                shuffle=False,
                seed=self.seed,
            )
        if stage in (None, "fit", "validate") and self.val_h5ad is not None and self._val is None:
            adata = ad.read_h5ad(self.val_h5ad)
            self._val = _PerturbseqDataset(
                adata,
                self.vocabulary_encoder,
                chunk_size=self.test_batch_size,
                shuffle=False,
                seed=self.seed,
            )

    def train_dataloader(self) -> DataLoader[dict[str, np.ndarray]]:
        """Return the DataLoader for the training dataset."""
        assert self._train is not None, "call setup(stage='fit') before train_dataloader()"
        return DataLoader(
            self._train,
            batch_size=1,
            shuffle=True,
            collate_fn=_collate,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader[dict[str, np.ndarray]]:
        """Return the DataLoader for the test dataset used for prediction."""
        assert self._test is not None, "call setup() before predict_dataloader()"
        return DataLoader(
            self._test,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[dict[str, np.ndarray]]:
        """Return the DataLoader for the dedicated validation dataset."""
        assert self._val is not None, "call setup() before val_dataloader()"
        return DataLoader(
            self._val,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate,
            num_workers=self.num_workers,
        )
