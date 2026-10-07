"""Utility classes and functions for CPA."""

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_array as SparseVector
from scipy.sparse import csr_matrix as SparseMatrix
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)


class MLP(nn.Module):
    """Multi-layer Perceptron with configurable number of layers, hidden dimension, dropout, and normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float | None = None,
        norm: str | None = "layer",
        elementwise_affine: bool = False,
    ):
        """Class for defining MLP with arbitrary number of layers."""
        super().__init__()  # type: ignore[no-untyped-call]

        if norm not in ["layer", "batch", None]:
            raise ValueError("norm must be one of ['layer', 'batch', None']")

        layers = nn.Sequential()
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(0, n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.network(x)


def ensure_2d_batch_tensor(value: Any) -> torch.Tensor | None:
    """Convert batch-like input to a tensor while preserving the leading batch axis."""
    if value is None:
        return None

    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


class Batch(NamedTuple):
    """Single Cell Expression Batch."""

    gene_expression: SparseMatrix
    perturbations: Sequence[list[str]] | SparseMatrix
    covariates: dict[str, Sequence[str] | SparseMatrix] | None = None
    controls: SparseMatrix | None = None
    id: Sequence[str] | None = None
    gene_names: Sequence[str] | None = None
    embeddings: np.ndarray | None = None


def get_covariates(
    df: pd.DataFrame, covariate_keys: list[str]
) -> tuple[dict[str, Sequence[str]], dict[str, list[Any]]]:
    """
    Get covariates from a dataframe.

    Args:
        df: a dataframe containing covariates for each cell with n_cells rows
        covariate_keys: a list of covariate keys in the dataframe

    Returns:
        A tuple (covariates, covariate_unique_values), where covariates is a
          dictionary of covariate keys to covariate values with n_cells rows,
          and covariate_unique_values is a dictionary of covariate keys to
          unique covariate values.

    Raises:
        KeyError: if a covariate key is not found in the dataframe.
    """
    try:
        covariates: dict[str, Sequence[str]] = {
            cov: df[cov].astype(str).tolist() for cov in covariate_keys
        }
        covariate_unique_values = {cov: list(df[cov].unique()) for cov in covariate_keys}
    except KeyError as e:
        raise KeyError("Covariate key not found in dataframe: " + str(e)) from e
    return covariates, covariate_unique_values


def build_control_dict(
    control_covariate_df: pd.DataFrame,
    covariate_keys: list[str] | None = None,
):
    """
    Build a dictionary of controls for each covariate condition.

    Args:
        control_covariate_df: pandas dataframe containing the covariates for each sample/cell
        covariate_keys: a list of keys in adata.obs that contain the covariates

    Returns:
        A dictionary of controls, where each control is a sparse matrix of size
          (n_controls_per_condition, n_genes) and the keys are
          dict[covariate_key, covariate_value].
    """
    if covariate_keys is None:
        covariate_keys = control_covariate_df.columns.tolist()

    grouped = control_covariate_df.groupby(
        list(covariate_keys), observed=True
    )  # groupby requires a list
    control_indexes = FrozenDictKeyMap()
    for group_key, group_indices in grouped.indices.items():
        if len(covariate_keys) == 1:
            assert isinstance(group_key, (str, int))
            group_key = (group_key,)

        key = dict(zip(covariate_keys, group_key, strict=True))  # # type: ignore[call-overload]
        control_indexes[key] = group_indices  # type: ignore[index]

    return control_indexes


def batch_dataloader(
    dataset: torch.utils.data.Dataset[Any],
    batch_size: int,
    shuffle: bool = True,
    oversample: bool = False,
    oversample_root: float = 2.0,
    **kwargs: object,
) -> DataLoader[Any]:
    """
    Build a PyTorch DataLoader from a PyTorch Dataset using a BatchSampler.

    Args:
        dataset: a PyTorch Dataset
        batch_size: the batch size
        shuffle: whether to shuffle the data
        oversample: whether to oversample the data
        oversample_root: oversampling weight will be
          `(1/class_frac)^(1/oversample_root)`
        kwargs: additional arguments to pass to DataLoaders
    """
    if oversample:
        assert oversample_root > 0, "Oversample root must be greater than 0"

        weights_dictionary: dict[str, float] = {}
        covariates_df = pd.DataFrame(dataset.covariates)  # type: ignore[assignment]  # convert dict of covariate tensors to dataframe for easier manipulation
        covariates_df["concatenated"] = covariates_df.apply(  # type: ignore[union-attr]  # apply function to each row to concatenate covariate values into a single string key
            lambda row: "_".join(row.values.astype(str)),  # type: ignore[union-attr]  # concatenate covariate values into a single string key
            axis=1,
        )
        covariate_fractions = covariates_df["concatenated"].value_counts() / covariates_df.shape[0]
        for covariate, frac in covariate_fractions.items():
            weight = (1 / frac) ** (1.0 / oversample_root)
            weights_dictionary[str(covariate)] = weight

        weights: list[float] = []
        dataset_length = len(covariates_df)
        for i in range(0, dataset_length):
            cov_values = [str(values[i]) for values in dataset.covariates.values()]  # type: ignore[attr-defined,union-attr]
            cov_key = "_".join(cov_values)
            weight = weights_dictionary[cov_key]
            weights.append(weight)

        sampler = WeightedRandomSampler(weights, dataset_length, replacement=True)

    elif shuffle:
        sampler = RandomSampler(dataset)  # type: ignore[arg-type]

    else:
        sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, sampler=batch_sampler, **kwargs)  # type: ignore[call-arg]
    return dataloader


class Example(NamedTuple):
    """Single Cell Expression Example."""

    # A vector of size (num_genes, )
    gene_expression: SparseVector
    # A list of perturbations applied to the cell
    perturbations: Sequence[str]  ## TODO: Should be [] if control
    # A map from covariate name to covariate value
    covariates: dict[str, str] | None = None
    # A map from control condition name to control gene expression of
    # shape (num_controls_in_condition, num_genes)
    controls: SparseVector | None = None
    # A cell id
    id: str | None = None
    # A list of gene names of length num_genes
    gene_names: Sequence[str] | None = None
    # Optional foundation model embeddings
    embeddings: np.ndarray | None = None


class FrozenDictKeyMap(dict[frozenset[tuple[str, Any]], Any]):
    """
    A dictionary that uses dictionaries as keys.

    Dictionaries cannot be used directly as keys to another dictionary because
    they are mutable. As a result this class first converts the dictionary to a
    frozenset of key-value pairs before using it as a key. The underlying data
    is stored using the dictionary data structure and this class just modifies
    the accessor and mutator methods.

    Example:
        >>> d = FrozenDictKeyMap()
        >>> d[{"a": 1, "b": 2}] = 1
        >>> d[{"a": 1, "b": 2}] = 2
        >>> d[{"a": 1, "b": 2}] = 3
        >>> d
        {frozenset({('a', 1), ('b', 2)}): 3}

    Attributes: see dict class
    """

    def __init__(self, data: Sequence[tuple[dict[str, Any], Any]] | None = None):
        """
        Initialize the dictionary.

        Args:
            data: a sequence of (key, value) pairs to initialize the dictionary
        """
        if data is not None:
            try:
                _data = [(frozenset(key.items()), value) for key, value in data]
            except AttributeError as exc:
                raise ValueError(
                    "data must be a sequence of (key, value) pairs where key is a dictionary"
                ) from exc
        else:
            _data = []
        super().__init__(_data)

    def __getitem__(self, key: dict[str, Any] | frozenset[tuple[str, Any]]) -> Any:  #
        """
        Get the value associated with the key.

        Args:
            key: a dictionary or a frozenset of key-value pairs.

        Returns:
            The value associated with the key.
        """
        if isinstance(key, frozenset):
            key = dict(key)
        return super().__getitem__(frozenset(key.items()))

    def __setitem__(self, key: dict[str, Any] | frozenset[tuple[str, Any]], value: Any) -> None:
        """
        Set the value associated with the key.

        Args:
            key: a dictionary.
            value: the value to set.
        """
        if isinstance(key, frozenset):
            key = dict(key)
        super().__setitem__(frozenset(key.items()), value)


def restore_perturbation_combinations(
    parsed_perturbations: Sequence[list[str]],
    delimiter: str | None = "+",
    perturbation_control_value: str | None = "control",
) -> pd.Series:
    """
    Restore the combined perturbations from a list of perturbations using a specified delimiter and control value.

    Args:
        parsed_perturbations: a sequence of lists of perturbations
        delimiter: a string that separates individual perturbations
        perturbation_control_value: a string that represents the control perturbation

    Returns:
        A pandas Series of combined perturbations
    """
    # combined_perturbations = []
    results: list[str | None] = []
    for combined_perts in parsed_perturbations:
        assert isinstance(combined_perts, list)
        pert_joined = (delimiter or "").join(combined_perts)
        if pert_joined == "":
            pert = perturbation_control_value
        else:
            pert = pert_joined
        results.append(pert)

    combined_perturbations = pd.Series(results, dtype="category")
    return combined_perturbations


def parse_perturbation_combinations(
    combined_perturbations: pd.Series,
    delimiter: str | None = "+",
    perturbation_control_value: str | None = "control",
) -> tuple[list[list[str]], list[str]]:
    """
    Get all perturbations applied to each cell.

    Args:
        combined_perturbations: combined perturbations string representation of
          size (n_cells, )
        delimiter: a string that separates individual perturbations
        perturbation_control_value: a string that represents the control perturbation

    Returns:
        A tuple (combinations, unique_perturbations), where combinations is an
          (n_cell, ) array of lists of individual perturbations applied to each
          cell, and unique_perturbations is a set of all unique perturbations.
    """
    assert isinstance(combined_perturbations.dtype, pd.CategoricalDtype)

    # Split the perturbations by the delimiter
    parsed: list[list[str]] = []
    uniques: dict[
        str, None
    ] = {}  ## Store unique perturbations as dictionary keys to ensure ordering is the same
    for combination in combined_perturbations.astype(str):
        perturbation_list: list[str] = []
        for perturbation in combination.split(delimiter):
            if perturbation != perturbation_control_value:
                perturbation_list.append(perturbation)
                uniques[perturbation] = None
        parsed.append(perturbation_list)

    return parsed, list(uniques.keys())


class noop_collate:
    """No operation collate function. Returns the batch as is."""

    def __call__(self, batch: list[Any]) -> Any:
        """Return the batch as is."""
        if len(batch) == 1:
            return batch[0]
        else:
            return batch
