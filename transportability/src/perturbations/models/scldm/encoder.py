"""Gene and label vocabulary encoder for scLDM models. Provides mapping from gene symbols and label categories to integer indices, handling metadata from AnnData objects or external JSON/parquet files."""

import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class VocabularyEncoderSimplified:
    """Encode a vocabulary of genes and labels into indices."""

    adata_path: Path | str | None
    class_vocab_sizes: dict[str, int]
    mask_token: str = "<MASK>"
    mask_token_idx: int = 0
    n_genes: int | None = None
    guidance_weight: dict[str, float] | None = None
    mu_size_factor: Path | str | dict[str, Any] | None = None
    sd_size_factor: Path | str | dict[str, Any] | None = None
    condition_strategy: Literal["mutually_exclusive", "joint"] = "mutually_exclusive"
    metadata_genes: Path | str | pd.DataFrame | None = None
    metadata_json: Path | str | None = None

    _token2idx: dict[str, int] = field(init=False, repr=False)
    _idx2token: dict[int, str] = field(init=False, repr=False)
    adata: ad.AnnData | None = field(init=False, default=None, repr=False)
    genes: np.ndarray = field(init=False, repr=False)
    gene_symbol_to_ensembl: dict[str, str] = field(init=False, repr=False)
    labels: dict[str, list[str]] | None = field(init=False, repr=False)
    _gene_token2idx: dict[str, int] = field(init=False, repr=False)
    _gene_idx2token: dict[int, str] = field(init=False, repr=False)
    gene_tokens_idx: list[int] = field(init=False, repr=False)
    classes2idx: dict[str, dict[str, int]] = field(init=False, repr=False)
    idx2classes: dict[str, dict[int, str]] = field(init=False, repr=False)
    joint_key: str = field(init=False, repr=False)
    joint_components: list[str] = field(init=False, repr=False)
    size_factor_marginal_key: str = field(init=False, repr=False)
    joint_idx_2_classes: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the encoder by loading metadata and setting up gene and label mappings."""
        metadata_payload = None
        if self.metadata_json is not None:
            metadata_path = Path(self.metadata_json)
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata_payload = json.load(f)

        if self.adata_path is not None and metadata_payload is None:
            self.adata = ad.read_h5ad(self.adata_path)
        else:
            self.adata = None
        if self.metadata_genes is not None:
            self.metadata_genes = pd.read_parquet(cast("Path | str", self.metadata_genes))
            self.genes = np.asarray(self.metadata_genes["feature_id"].values)
            # Create conversion dict from gene symbols to ensemble genes
            self.gene_symbol_to_ensembl = dict(
                zip(
                    self.metadata_genes["feature_name"].values,
                    self.metadata_genes["feature_id"].values,
                    strict=False,
                )
            )
        elif metadata_payload is not None:
            self.genes = np.asarray(metadata_payload["genes"])
        else:
            assert self.adata is not None
            self.genes = np.asarray(self.adata.var_names.values)

        # Auto-detect n_genes if not provided or mismatched
        detected_n_genes = len(self.genes)
        if self.n_genes is None:
            self.n_genes = detected_n_genes
        elif self.n_genes != detected_n_genes:
            # Prefer the adata var_names length to avoid runtime failures
            self.n_genes = detected_n_genes

        if self.adata is not None:
            self.labels = {
                label: self.adata.obs[label].cat.categories.tolist()
                for label in self.class_vocab_sizes.keys()
            }
        elif metadata_payload is not None:
            label_payload = metadata_payload.get("labels", {})
            self.labels = {}
            for label in self.class_vocab_sizes.keys():
                if label not in label_payload:
                    raise ValueError(f"metadata_json missing label categories for '{label}'")
                self.labels[label] = label_payload[label]
        else:
            self.labels = None

        genes_tokens = ["<MASK>"]
        genes_tokens += list(self.genes)

        self._gene_token2idx = {token: idx for idx, token in enumerate(map(str, genes_tokens))}
        self._gene_idx2token = dict(enumerate(genes_tokens))

        self.gene_tokens_idx = list(self._gene_token2idx.values())[1:]
        assert self.mask_token_idx == self._gene_token2idx[self.mask_token]

        if self.labels is not None:
            self.classes2idx = {
                label: {token: idx for idx, token in enumerate(map(str, self.labels[label]))}
                for label in self.class_vocab_sizes.keys()
            }
            self.idx2classes = {
                label: {idx: token for token, idx in self.classes2idx[label].items()}
                for label in self.class_vocab_sizes.keys()
            }

        # size factors
        if hasattr(self, "condition_strategy") and self.condition_strategy != "joint":
            if self.mu_size_factor is not None:
                with Path(cast("Path | str", self.mu_size_factor)).open("rb") as f:
                    mu_size_factor_dict = pickle.load(f)
                self.mu_size_factor = {}
                for label in self.class_vocab_sizes.keys():
                    self.mu_size_factor[label] = {
                        self.classes2idx[label][k]: v for k, v in mu_size_factor_dict[label].items()
                    }

            if self.sd_size_factor is not None:
                with Path(cast("Path | str", self.sd_size_factor)).open("rb") as f:
                    sd_size_factor_dict = pickle.load(f)
                self.sd_size_factor = {}
                for label in self.class_vocab_sizes.keys():
                    self.sd_size_factor[label] = {
                        self.classes2idx[label][k]: v for k, v in sd_size_factor_dict[label].items()
                    }
        elif hasattr(self, "condition_strategy") and self.condition_strategy == "joint":
            joint_class = "_".join(self.class_vocab_sizes.keys())
            self.joint_key = joint_class
            self.joint_components = list(self.class_vocab_sizes.keys())
            # Marginal used to fall back the size factor when an unseen (cell_line, gene) joint
            # combination is generated: the cell_line marginal (first joint component).
            self.size_factor_marginal_key = self.joint_components[0]
            if self.mu_size_factor is not None:
                with Path(cast("Path | str", self.mu_size_factor)).open("rb") as f:
                    mu_size_factor_dict = pickle.load(f)
                self.mu_size_factor = {}
                self.mu_size_factor[joint_class] = mu_size_factor_dict[joint_class]
                self.joint_idx_2_classes = {}
                class1, class2 = self.class_vocab_sizes.keys()
                for _idx, token in enumerate(mu_size_factor_dict[joint_class].keys()):
                    # get class instances from token (use rsplit to handle underscores in instance names)
                    instance1, instance2 = token.rsplit("_", 1)

                    class1_idx = self.classes2idx[class1][instance1]
                    class2_idx = self.classes2idx[class2][instance2]
                    self.joint_idx_2_classes[str(class1_idx) + "_" + str(class2_idx)] = token
                # Per-cell_line marginal log-size-factor means, keyed by class index.
                marginal_key = self.size_factor_marginal_key
                if marginal_key in mu_size_factor_dict:
                    self.mu_size_factor[marginal_key] = {
                        self.classes2idx[marginal_key][k]: v
                        for k, v in mu_size_factor_dict[marginal_key].items()
                    }

            if self.sd_size_factor is not None:
                with Path(cast("Path | str", self.sd_size_factor)).open("rb") as f:
                    sd_size_factor_dict = pickle.load(f)
                self.sd_size_factor = {}
                self.sd_size_factor[joint_class] = sd_size_factor_dict[joint_class]
                # Per-cell_line marginal log-size-factor stds, keyed by class index.
                marginal_key = self.size_factor_marginal_key
                if marginal_key in sd_size_factor_dict:
                    self.sd_size_factor[marginal_key] = {
                        self.classes2idx[marginal_key][k]: v
                        for k, v in sd_size_factor_dict[marginal_key].items()
                    }

            # handle idx mapping later during generation time
        # Remove adata reference as it's no longer needed after initialization
        del self.adata
        self.adata = None

    def encode_genes(self, tokens: Sequence[str]) -> np.ndarray:
        """
        Convert tokens to their corresponding indices.

        Ensures a numeric dtype output. Unknown tokens map to the mask token index.
        """
        mask_idx = self.mask_token_idx
        indices = [self._gene_token2idx.get(str(token), mask_idx) for token in tokens]
        return np.asarray(indices, dtype=np.int64)

    def decode_genes(self, indices: Sequence[int]) -> np.ndarray:
        """Convert indices back to their corresponding tokens."""

        def _lookup(idx: int) -> str | None:
            return self._gene_idx2token.get(idx, None)

        return np.vectorize(_lookup)(indices)

    def encode_metadata(self, metadata: Sequence[str] | np.ndarray, label: str) -> np.ndarray:
        """
        Convert metadata items to their corresponding indices for a given label.

        Unknown items map to None.
        """
        return np.array([self.classes2idx[label].get(str(item), None) for item in metadata])

    def decode_metadata(self, indices: Sequence[int], label: str) -> np.ndarray:
        """
        Convert indices back to their corresponding metadata items for a given label.

        Unknown indices map to None.
        """
        return np.array([self.idx2classes[label].get(item, None) for item in indices])
