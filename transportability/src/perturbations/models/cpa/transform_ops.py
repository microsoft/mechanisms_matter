"""
Transform operations for the CPA model.

This module defines various transform operations that can be applied to the input data for the CPA model. These transforms include converting sparse matrices to dense, encoding categorical variables as one-hot vectors, and encoding multi-label perturbations as binary vectors. The transforms are designed to be composable and can be applied to individual examples or batches of data. The SingleCellPipeline class provides a convenient way to apply a sequence of transforms to the input data, including the appropriate transforms for perturbations and covariates based on the training context. The transforms are used in the data loading and preprocessing steps of the CPA model, and they ensure that the input data.
"""

import functools
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence
from typing import Any, TypeVar

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from .utils import Batch, Example

Datum = TypeVar("Datum", Example, Batch)

# Perturbations use MultiLabelBinarizer because a cell can be subject to multiple perturbations simultaneously (combinatorial perturbations).
# Covariates use standard OneHotEncoder because each covariate has a single value per cell

# Type aliases
ExampleMultiLabel = list[str]
BatchMultiLabel = list[list[str]]


class Transform(ABC):
    """Abstract transform interface."""

    @abstractmethod
    def __call__(self, data: Datum) -> Datum:
        """Apply the transform to a datum."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return self.__class__.__name__ + "({!s})"


class Compose(list[Transform], Transform):
    """Creates a transform from a sequence of transforms."""

    def __call__(self, data: Datum) -> Datum:
        """Apply the sequence of transforms to the data."""
        for transform in self:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        """Return a string representation of the sequence of transforms."""
        transforms_repr = " \u2192 ".join(repr(transform) for transform in self)
        return f"[{transforms_repr}]"


class ToDense(Transform):
    """Convert a sparse matrix/tensor to a dense matrix/tensor."""

    def __call__(self, value: Any) -> torch.Tensor:  # type: ignore[override]
        """Convert a sparse matrix/tensor to a dense matrix/tensor."""
        if isinstance(value, torch.Tensor):
            return value.to_dense()
        elif isinstance(value, csr_matrix):
            return torch.Tensor(value.toarray())  # type: ignore
        else:
            return value

    def __repr__(self):
        """Return a string representation of the ToDense transform."""
        return "ToDense"


class ToFloat(Transform):
    """Convert a tensor to float."""

    def __call__(self, value: Any) -> torch.Tensor:  # type: ignore[override]
        """Convert a tensor to float."""
        if isinstance(value, torch.Tensor):
            return value.float()
        else:
            return value

    def __repr__(self):
        """Return a string representation of the ToFloat transform."""
        return "ToFloat"


class MapApply(Transform):
    """
    Map each transform to an input based on a key.

    Attributes:
        transform_map: A map of key to transform.
    """

    transform_map: dict[str, Transform | Callable[..., Any]]

    def __init__(
        self,
        transforms: dict[str, Transform | Callable[..., Any]],
        init_params_map: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the instance based on passed transforms.

        This classes supports two ways of initializing the transforms. The first
        is by passing a map of key to transform. The second is by passing a map of
        key to factory callable. The factory callable will be called with the
        corresponding init params from the init_params_map. The factory callable
        should return a Transform.

        Args:
            transforms: A map of key to transform.
            init_params_map: A map of key to init params for the transforms.

        Raises:
            ValueError: If init_params_map is not None when using a dict of
                Transforms.
            TypeError: If the transform is not a dict of Transform or a callable.
        """
        super().__init__()
        self.transform_map = {}
        for key, transform in transforms.items():
            # Transforms are dict[str, Transform], directly assign them
            if isinstance(transform, Transform):
                if init_params_map is not None:
                    raise ValueError(
                        "init_params_map should be None when using a dict of Transforms."
                    )
                self.transform_map[key] = transform
            # Transforms are dict[str, factory_callable], call the factory
            elif callable(transform):
                if init_params_map is None:
                    raise ValueError(
                        "init_params_map must be provided when using factory callables."
                    )
                self.transform_map[key] = transform(init_params_map[key])
            else:
                raise TypeError(
                    f"Invalid type for {key=} in transform. Must be either a Transform or a callable."
                )

    def __call__(self, value_map: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Apply the corresponding transform to each value based on the key."""
        return {key: self.transform_map[key](val) for key, val in value_map.items()}

    def __repr__(self) -> str:
        """Return a string representation of the MapApply transform."""
        transforms_repr = ", ".join(
            f"{key}: {transform!r}" for key, transform in self.transform_map.items()
        )
        return "{" + transforms_repr + "}"


class OneHotEncode(Transform):
    """
    One-hot encode a categorical variable.

    Attributes:
        one_hot_encoder: the wrapped encoder instance
    """

    one_hot_encoder: OneHotEncoder

    def __init__(self, categories: Collection[str], **kwargs: Any):
        """Initialize the one-hot encoder."""
        cat_list = [list(categories)]
        self.one_hot_encoder = OneHotEncoder(
            categories=cat_list,
            sparse_output=False,
            **kwargs,
        )

    def __call__(self, labels: Sequence[str]):  # type: ignore[override]
        """Encode the labels as one-hot tensors."""
        string_array: np.ndarray = np.array(labels).reshape(-1, 1)  # type: ignore
        encoded = self.one_hot_encoder.fit_transform(string_array)  # type: ignore
        return torch.Tensor(encoded)

    def __repr__(self):
        """Return a string representation of the one-hot encoder."""
        _base = super().__repr__()
        categories = ", ".join(self.one_hot_encoder.categories[0])  # type: ignore[index]
        return _base.format(categories)


class Dispatch(dict[str, Any], Transform):
    """
    Dispatches a transform to an example based on a key field.

    Attributes:
        self: A map of key to transform.
    """

    def __call__(self, data: Example | Batch) -> Example | Batch:  # type: ignore[override]
        """Apply each transform to the field of an example matching its key."""
        result: dict[str, Any] = {}
        for key, transform in self.items():
            try:
                result[key] = transform(getattr(data, key))
            except (KeyError, AttributeError) as exc:
                raise TypeError(
                    f"Invalid key '{key}' in transforms. All keys need to match the fields of an example."
                ) from exc
        return data._replace(**result)

    def __repr__(self) -> str:
        """Return a string representation of the dispatch transform."""
        _base = Transform.__repr__(self)
        transforms_repr = ", ".join(f"{key}: {transform!r}" for key, transform in self.items())
        return _base.format(transforms_repr)


class SingleCellPipeline(Dispatch):
    """Single cell transform pipeline."""

    def __init__(
        self,
        perturbation_uniques: set[str],
        covariate_uniques: dict[str, set[str]],
    ) -> None:
        """Initialize the pipeline with the appropriate transforms for perturbations and covariates."""
        # Set up covariates transform
        covariate_transform = {
            key: Compose([OneHotEncode(uniques), ToFloat()])
            for key, uniques in covariate_uniques.items()
        }
        # Initialize the pipeline
        super().__init__(
            perturbations=Compose(
                [
                    MultiLabelEncode(perturbation_uniques),
                    ToFloat(),
                ]
            ),
            gene_expression=ToDense(),
            covariates=MapApply(covariate_transform),
        )


class MultiLabelEncode(Transform):
    """
    Transforms a sequence of labels into a binary vector.

    Attributes:
        label_binarizer: the wrapped binarizer instance

    Raises:
        ValueError: if any of the labels are not found in the encoder classes
    """

    label_binarizer: MultiLabelBinarizer

    def __init__(self, classes: Collection[str]):
        """Initialize the multi-label encoder."""
        self.label_binarizer = MultiLabelBinarizer(classes=list(classes), sparse_output=False)

    @functools.cached_property
    def classes(self) -> set[str]:
        """Return the set of classes."""
        # return set(self.label_binarizer.classes)
        return set(str(c) for c in self.label_binarizer.classes)  # type: ignore[attr-defined]

    def __call__(self, labels: ExampleMultiLabel | BatchMultiLabel) -> torch.Tensor:  # type: ignore[override]
        """Encode the labels as binary vectors."""
        # If labels is a single example, convert it to a batch
        if not labels or isinstance(labels[0], str):
            batch_labels: BatchMultiLabel = [labels]  # type: ignore[list-item]
        else:
            batch_labels = labels  # type: ignore
        self._check_inputs(batch_labels)
        encoded = self.label_binarizer.fit_transform(batch_labels)  # type: ignore[arg-type]
        return torch.from_numpy(encoded)  # type: ignore

    def _check_inputs(self, labels: BatchMultiLabel):
        unique_labels = set(itertools.chain.from_iterable(labels))
        if not unique_labels <= self.classes:
            missing_labels = unique_labels - self.classes
            raise ValueError(
                f"Labels {missing_labels} not found in the encoder classes {self.classes}"
            )

    def __repr__(self):
        """Return a string representation of the multi-label encoder."""
        _base = super().__repr__()
        classes = ", ".join(str(c) for c in self.label_binarizer.classes)  # type: ignore[attr-defined]
        return _base.format(classes)
