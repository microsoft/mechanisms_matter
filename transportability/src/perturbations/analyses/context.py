"""Context-aware split planning utilities for perturbation analyses."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

DEFAULT_CONTEXT_AXIS = "cell_line"
CONTEXT_DATA_MAP = {
    "norman19": {
        "context_axis": "cell_line",
        "heldout_values": None,
    },  # Single context (cell line), so no held-out context values.
    "replogle22": {
        "context_axis": "cell_line",
        "heldout_values": ["K562"],
    },  # Held-out cell line for cross-context evaluation.
    "CD4+": {
        "context_axis": "donor_timepoint",
        "heldout_values": [
            "D2_Rest",
            "D2_Stim8hr",
            "D2_Stim48hr",
            "D3_Rest",
            "D3_Stim8hr",
            "D3_Stim48hr",
        ],
    },
}
IN_CONTEXT_FRACTIONS = np.asarray([0.5, 0.2, 0.3], dtype=np.float64)


@dataclass(frozen=True)
class StrategySplit:
    """Container of train/validation/test index arrays for one strategy."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class DatasetContextConfig:
    """Context-axis and held-out values for one dataset."""

    context_axis: str = DEFAULT_CONTEXT_AXIS
    heldout_values: tuple[Any, ...] | None = None


def get_dataset_context_config(dataset_name: str) -> DatasetContextConfig:
    """Return validated context configuration for one real dataset."""
    if dataset_name not in CONTEXT_DATA_MAP:
        raise KeyError(
            f"Missing context configuration for dataset '{dataset_name}'. "
            f"Available datasets: {sorted(CONTEXT_DATA_MAP)}"
        )

    config = CONTEXT_DATA_MAP[dataset_name]
    if not isinstance(config, dict):
        raise TypeError(
            f"Context configuration for dataset '{dataset_name}' must be a dict. "
            f"Got {type(config).__name__}."
        )
    context_axis = config.get("context_axis")
    heldout_values = config.get("heldout_values")
    return DatasetContextConfig(
        context_axis=DEFAULT_CONTEXT_AXIS if context_axis is None else str(context_axis),
        heldout_values=None if heldout_values is None else tuple(heldout_values),
    )


@dataclass(frozen=True)
class EvaluationContext:
    """One evaluation slice keyed by context axis and value tuple."""

    axis: str
    values: tuple[Any, ...]

    @property
    def value_label(self) -> str:
        """Return a printable label for context values."""
        if len(self.values) == 0:
            return "all"
        return "|".join(str(value) for value in self.values)


@dataclass(frozen=True)
class SplitMetadata:
    """Metadata describing how a split plan was generated."""

    seed: int
    context_axis: str
    context_values: tuple[Any, ...]
    test_context_values: tuple[Any, ...]
    held_out_perturbations: tuple[Any, ...]
    evaluation_contexts: tuple[EvaluationContext, ...]


@dataclass(frozen=True)
class SplitPlan:
    """Pair of in-context and cross-context splits with shared metadata."""

    metadata: SplitMetadata
    in_context: StrategySplit
    cross_context: StrategySplit


def _stable_unique(values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(list(values)))


def deterministic_perturbation(
    values: Sequence[Any] | np.ndarray, seed: int, key: Any | None = None
) -> np.ndarray:
    """
    Return a deterministic permutation independent of prior RNG draws.

    The optional ``key`` lets different split steps derive independent shuffles
    while remaining reproducible for the same ``seed`` and logical selection.
    """
    payload = f"{seed}|{key!r}".encode()
    derived_seed = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), byteorder="little"
    )
    rng = np.random.default_rng(derived_seed)
    return np.asarray(rng.permutation(values))


def build_evaluation_contexts(
    splitter: ContextSplitter,
    seed: int | None = None,
) -> list[EvaluationContext]:
    """Build evaluation context descriptors from splitter metadata."""
    metadata = splitter.get_split_metadata(seed=seed)
    return list(metadata.evaluation_contexts)


class ContextSplitter:
    """Seeded splitter with shared in-context/cross-context planning."""

    def __init__(
        self,
        adata: Any,
        split_strategy: Literal["in-context", "cross-context"] = "in-context",
        perturbation_key: str = "perturbation",
        context_axis: str | None = None,
        test_context_values: Sequence[Any] | None = None,
        control_label: Any | Sequence[Any] | None = "control",
    ) -> None:
        """Initialize a context-aware splitter over `adata.obs`."""
        self.perturbation_key = perturbation_key
        self.split_strategy = split_strategy
        self.control_label = control_label

        self.obs = adata.obs.copy()
        self.n_obs = adata.n_obs
        assert self.perturbation_key in self.obs.columns, (
            f"Missing perturbation column '{self.perturbation_key}' in obj.obs."
        )

        resolved_context_axis = (
            str(context_axis) if context_axis is not None else DEFAULT_CONTEXT_AXIS
        )
        if resolved_context_axis not in self.obs.columns:
            raise KeyError(
                f"Context axis '{resolved_context_axis}' not found in obj.obs. "
                f"Available columns: {list(self.obs.columns)}"
            )

        self.context_axis = resolved_context_axis
        self.context_values = _stable_unique(self.obs[self.context_axis].to_numpy(copy=False))
        self.test_context_values = tuple(dict.fromkeys(list(test_context_values or [])))
        self._perturbations = _stable_unique(self.obs[self.perturbation_key].to_numpy(copy=False))
        self._control_values = tuple(
            self._fetch_controls(np.asarray(self._perturbations, dtype=object)).tolist()
        )
        self._non_control_perturbations = tuple(
            pert for pert in self._perturbations if pert not in set(self._control_values)
        )
        self._plan_cache: dict[int, SplitPlan] = {}
        self.last_split_metadata: SplitMetadata | None = None
        self.held_out_perturbations: tuple[Any, ...] = ()
        self.evaluation_context_values: tuple[Any, ...] = ()

        self._check_split_configuration()

    def split(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return train/val/test indices for the configured split strategy."""
        plan = self._build_split_plan(seed=seed)
        self.last_split_metadata = plan.metadata
        self.held_out_perturbations = plan.metadata.held_out_perturbations
        self.evaluation_context_values = tuple(
            context.values[0]
            for context in plan.metadata.evaluation_contexts
            if len(context.values) == 1
        )

        if self.split_strategy == "in-context":
            split = plan.in_context
        else:
            split = plan.cross_context

        self._assert_disjoint(split.train_idx, split.val_idx, split.test_idx)
        return split.train_idx, split.val_idx, split.test_idx

    def get_split_metadata(self, seed: int | None = None) -> SplitMetadata:
        """Return split metadata from cache or build it for a given seed."""
        if seed is None:
            if self.last_split_metadata is None:
                raise ValueError(
                    "No split metadata available. Call split(seed) first or provide a seed."
                )
            return self.last_split_metadata
        return self._build_split_plan(seed=seed).metadata

    def _check_split_configuration(self) -> None:
        if self.split_strategy not in ("in-context", "cross-context"):
            raise ValueError(
                "split_strategy must be one of 'in-context' or 'cross-context'. "
                f"Got {self.split_strategy!r}."
            )

        if len(self.context_values) == 0:
            raise ValueError(f"No context values found in '{self.context_axis}'.")

        if len(self.test_context_values) == 0:
            if len(self.context_values) == 1:
                self.test_context_values = (self.context_values[0],)
            else:
                raise ValueError(
                    "Datasets with multiple context values require non-empty test_context_values."
                )

        missing = [
            value for value in self.test_context_values if value not in set(self.context_values)
        ]
        if missing:
            raise ValueError(
                f"test_context_values contains unknown value(s) for '{self.context_axis}': {missing}. "
                f"Available values include: {sorted(self.context_values, key=lambda x: str(x))}"
            )

        if len(self.context_values) == 1:
            self.test_context_values = (self.context_values[0],)
            if self.split_strategy == "cross-context":
                raise ValueError(
                    "Cross-context split requires at least two distinct context values."
                )

    @staticmethod
    def _rounded_partition_counts(
        n_total: int,
        fractions: np.ndarray,
    ) -> tuple[int, ...]:
        if n_total == 0:
            return tuple(0 for _ in range(len(fractions)))

        fractions = np.asarray(fractions, dtype=np.float64)
        if np.any(fractions < 0):
            raise ValueError("fractions must be non-negative.")
        total_fraction = fractions.sum()
        if total_fraction <= 0:
            raise ValueError("At least one fraction must be positive.")

        normalized = fractions / total_fraction
        targets = normalized * n_total
        counts = np.rint(targets).astype(np.int64)
        delta = n_total - counts.sum()

        if delta > 0:
            priority = np.argsort(-(targets - counts), kind="stable")
            for i in range(delta):
                counts[int(priority[i % counts.size])] += 1
        elif delta < 0:
            priority = np.argsort(-(counts - targets), kind="stable")
            to_remove = -delta
            ptr = 0
            while to_remove > 0:
                idx = int(priority[ptr % counts.size])
                if counts[idx] > 0:
                    counts[idx] -= 1
                    to_remove -= 1
                ptr += 1

        if counts.sum() != n_total or np.any(counts < 0):
            raise RuntimeError(
                f"Failed to partition n_total={n_total} with fractions={normalized.tolist()}."
            )

        return tuple(count for count in counts.tolist())

    def _group_indices_by_context_and_perturbation(
        self,
    ) -> dict[tuple[Any, Any], np.ndarray]:
        grouped = self.obs.groupby(
            [self.context_axis, self.perturbation_key],
            sort=False,
            observed=False,
        ).indices
        return {
            key: np.asarray(grouped[key], dtype=np.int64)
            for key in sorted(grouped.keys(), key=lambda x: repr(x))
        }

    def _build_split_plan(self, seed: int) -> SplitPlan:
        cached = self._plan_cache.get(seed)
        if cached is not None:
            return cached

        context_arr = self.obs[self.context_axis].to_numpy(copy=False)
        perturbation_arr = self.obs[self.perturbation_key].to_numpy(copy=False)
        if self._control_values:
            control_mask = np.isin(perturbation_arr, self._control_values)
        else:
            control_mask = np.zeros(self.n_obs, dtype=bool)
        grouped = self._group_indices_by_context_and_perturbation()

        # Start with in-context splitting, get a sense of the numbers for each context/perturbation group.
        # This will inform the cross-context split to ensure comparable test sizes.
        in_train: list[int] = []
        in_val: list[int] = []
        in_test: list[int] = []
        raw_in_context_test_by_group: dict[tuple[Any, Any], np.ndarray] = {}

        for group_key, idx in grouped.items():
            if idx.size == 0:
                raw_in_context_test_by_group[group_key] = np.asarray([], dtype=np.int64)
                continue

            shuffled = deterministic_perturbation(idx, seed=seed, key=("group", group_key))
            n_train, n_val, _ = self._rounded_partition_counts(shuffled.size, IN_CONTEXT_FRACTIONS)
            group_test = np.asarray(shuffled[n_train + n_val :], dtype=np.int64)
            raw_in_context_test_by_group[group_key] = group_test

            in_train.extend(shuffled[:n_train].tolist())
            in_val.extend(shuffled[n_train : n_train + n_val].tolist())
            in_test.extend(group_test.tolist())

        raw_in_context_train = np.asarray(in_train, dtype=np.int64)
        raw_in_context_val = np.asarray(in_val, dtype=np.int64)
        raw_in_context_test = np.asarray(in_test, dtype=np.int64)
        self._assert_disjoint(raw_in_context_train, raw_in_context_val, raw_in_context_test)

        # Determine which perturbations to hold out for cross-context testing.
        # Then apply the same perturbation for in-context test.
        # This ensures that the in-context and cross-context test sets are comparable in terms of perturbation types.
        if len(self.context_values) > 1 and len(self._non_control_perturbations) > 0:
            shuffled_perturbations = deterministic_perturbation(
                self._non_control_perturbations,
                seed=seed,
                key="held_out_perturbations",
            )
            n_holdout = int(
                np.ceil(shuffled_perturbations.size / 2)
            )  # hold out half of the perturbations for cross-context test
            held_out_perturbations = tuple(shuffled_perturbations[:n_holdout].tolist())
        else:
            held_out_perturbations = ()

        if len(self.context_values) > 1:
            held_out_context_mask = np.isin(context_arr, self.test_context_values)
            held_out_context_perturbation_mask = control_mask | np.isin(
                perturbation_arr,
                held_out_perturbations,
            )
            # Keep the full raw in-context test split outside the held-out contexts.
            # Within held-out contexts, keep only controls plus the perturbations
            # selected for cross-context holdout/alignment.
            in_context_keep_mask = (~held_out_context_mask) | (
                held_out_context_mask & held_out_context_perturbation_mask
            )
            final_in_context_test = raw_in_context_test[
                in_context_keep_mask[raw_in_context_test]
            ].astype(np.int64, copy=False)
        else:
            final_in_context_test = raw_in_context_test.astype(np.int64, copy=False)

        # Now build the cross-context split,
        # ensuring the train/val/test sizes are comparable to the in-context split.
        cross_train: list[int] = []
        cross_val: list[int] = []
        cross_test: list[int] = []
        assigned_mask = np.zeros(self.n_obs, dtype=bool)

        for context_value in self.test_context_values:
            control_idx = np.flatnonzero((context_arr == context_value) & control_mask).astype(
                np.int64,
                copy=False,
            )
            if control_idx.size == 0:
                continue

            control_key: Any
            if len(self._control_values) == 1:
                control_key = (context_value, self._control_values[0])
            else:
                control_key = ("controls", context_value)
            shuffled = deterministic_perturbation(
                control_idx, seed=seed, key=("group", control_key)
            )
            n_train, n_val, _ = self._rounded_partition_counts(shuffled.size, IN_CONTEXT_FRACTIONS)
            cross_train.extend(shuffled[:n_train].tolist())
            cross_val.extend(shuffled[n_train : n_train + n_val].tolist())
            cross_test.extend(shuffled[n_train + n_val :].tolist())
            assigned_mask[control_idx] = True

        held_out_test_candidates: dict[tuple[Any, Any], np.ndarray] = {}
        for context_value in self.test_context_values:
            for perturbation in held_out_perturbations:
                idx = np.flatnonzero(
                    (context_arr == context_value) & (perturbation_arr == perturbation)
                ).astype(np.int64, copy=False)
                held_out_test_candidates[(context_value, perturbation)] = idx
                if idx.size > 0:
                    assigned_mask[idx] = True

        remaining_idx = np.flatnonzero(~assigned_mask).astype(np.int64, copy=False)
        if remaining_idx.size > 0:
            shuffled_remaining = deterministic_perturbation(
                remaining_idx, seed=seed, key="remaining"
            )
            n_train, n_val = self._rounded_partition_counts(
                shuffled_remaining.size, IN_CONTEXT_FRACTIONS[:2]
            )
            cross_train.extend(shuffled_remaining[:n_train].tolist())
            cross_val.extend(shuffled_remaining[n_train : n_train + n_val].tolist())

        for group_key, idx in held_out_test_candidates.items():
            selected = raw_in_context_test_by_group.get(group_key, np.asarray([], dtype=np.int64))
            if selected.size == 0 or idx.size == 0:
                continue
            if selected.size > idx.size:
                raise RuntimeError(
                    f"Cross-context test downsampling target {selected.size} exceeds "
                    f"available cells ({idx.size}) for context {group_key!r}."
                )
            cross_test.extend(selected.tolist())

        raw_cross_train = np.asarray(cross_train, dtype=np.int64)
        raw_cross_val = np.asarray(cross_val, dtype=np.int64)
        raw_cross_test = np.asarray(cross_test, dtype=np.int64)
        self._assert_disjoint(raw_cross_train, raw_cross_val, raw_cross_test)

        if len(self.context_values) == 1:
            raw_cross_train = raw_in_context_train
            raw_cross_val = raw_in_context_val
            raw_cross_test = final_in_context_test

        train_target_size = min(raw_in_context_train.size, raw_cross_train.size)
        final_in_context_train = self._downsample_indices(
            indices=raw_in_context_train,
            target_size=train_target_size,
            seed=seed,
            key="train",
        )
        final_cross_train = self._downsample_indices(
            indices=raw_cross_train,
            target_size=train_target_size,
            seed=seed,
            key="train",
        )
        val_target_size = min(raw_in_context_val.size, raw_cross_val.size)
        final_in_context_val = self._downsample_indices(
            indices=raw_in_context_val,
            target_size=val_target_size,
            seed=seed,
            key="val",
        )
        final_cross_val = self._downsample_indices(
            indices=raw_cross_val,
            target_size=val_target_size,
            seed=seed,
            key="val",
        )

        evaluation_values = (
            self.test_context_values if len(self.context_values) > 1 else (self.context_values[0],)
        )
        evaluation_contexts = tuple(
            EvaluationContext(
                axis=self.context_axis,
                values=(value,),
            )
            for value in evaluation_values
        )

        metadata = SplitMetadata(
            seed=seed,
            context_axis=self.context_axis,
            context_values=self.context_values,
            test_context_values=evaluation_values,
            held_out_perturbations=held_out_perturbations,
            evaluation_contexts=evaluation_contexts,
        )
        plan = SplitPlan(
            metadata=metadata,
            in_context=StrategySplit(
                train_idx=final_in_context_train,
                val_idx=final_in_context_val,
                test_idx=final_in_context_test,
            ),
            cross_context=StrategySplit(
                train_idx=final_cross_train,
                val_idx=final_cross_val,
                test_idx=raw_cross_test,
            ),
        )
        self._plan_cache[seed] = plan
        return plan

    @staticmethod
    def _downsample_indices(
        indices: np.ndarray,
        target_size: int,
        seed: int,
        key: Any,
    ) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64)
        if target_size < 0:
            raise ValueError("target_size must be non-negative.")
        if index_array.size <= target_size:
            return index_array.astype(np.int64, copy=True)
        selected = deterministic_perturbation(index_array, seed=seed, key=("downsample", key))[
            :target_size
        ]
        return np.asarray(selected, dtype=np.int64)

    def _fetch_controls(self, perturbations: np.ndarray) -> np.ndarray:
        if self.control_label is not None:
            if isinstance(self.control_label, (str, bytes)):
                requested = np.asarray([self.control_label], dtype=object)
            else:
                try:
                    requested = np.asarray(list(self.control_label), dtype=object)
                except TypeError:
                    requested = np.asarray([self.control_label], dtype=object)

            matched = perturbations[np.isin(perturbations, requested)]
            if matched.size == 0:
                raise ValueError(
                    f"Requested control_label={self.control_label!r} was not found in "
                    f"obs['{self.perturbation_key}']."
                )
            return matched

        inferred_controls: list[Any] = []
        for value in perturbations:
            if isinstance(value, (np.integer, int, np.floating, float)) and value == -1:
                inferred_controls.append(value)
                continue
            if isinstance(value, str) and value.strip().lower() in {"control", "ctrl"}:
                inferred_controls.append(value)

        if not inferred_controls:
            return np.asarray([], dtype=object)
        return np.asarray(list(dict.fromkeys(inferred_controls)), dtype=object)

    def _assert_disjoint(
        self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
    ) -> None:
        assignment = np.zeros(self.n_obs, dtype=np.int8)
        for split_name, split_idx in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            if split_idx.size == 0:
                continue
            idx = np.asarray(split_idx, dtype=np.int64)
            if np.any((idx < 0) | (idx >= self.n_obs)):
                raise RuntimeError(f"{split_name} split contains out-of-range indices.")
            np.add.at(assignment, idx, 1)

        if np.any(assignment > 1):
            n_overlapping = np.sum(assignment > 1)
            raise RuntimeError(
                f"Split indices must be disjoint. Overlapping rows: {n_overlapping}."
            )


def filter_indices_to_context_values(
    obs: pd.DataFrame,
    indices: np.ndarray,
    context_axis: str | None,
    context_values: Sequence[Any],
) -> np.ndarray:
    """Filter row indices to those whose context value is in `context_values`."""
    index_array = np.asarray(indices, dtype=np.int64)
    if context_axis is None or context_axis not in obs.columns or len(context_values) == 0:
        return index_array.copy()

    context_arr = obs.iloc[index_array][context_axis].to_numpy(copy=False)
    return index_array[np.isin(context_arr, tuple(context_values))]


def indexer_from_labels(
    all_labels: Sequence[Any],
    selected_labels: Sequence[Any],
) -> np.ndarray:
    """Return positions of `selected_labels` within `all_labels`."""
    label_to_pos = {label: idx for idx, label in enumerate(list(all_labels))}
    return np.asarray([label_to_pos[label] for label in list(selected_labels)], dtype=np.int64)
