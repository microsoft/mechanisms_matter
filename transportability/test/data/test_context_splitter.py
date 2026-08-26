from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from perturbations.analyses.context import ContextSplitter, build_evaluation_contexts


class AnnDataProxy:
    """Minimal AnnData-like wrapper used by lightweight tests."""

    def __init__(self, obs: pd.DataFrame) -> None:
        """Initialize the proxy with a DataFrame representing observations."""
        self.obs = obs
        self.n_obs = len(obs)


def _make_obs(
    context_axis: str,
    context_values: list[object],
    perturbations: list[str],
    cells_per_group: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for context_value in context_values:
        for perturbation in perturbations:
            for _ in range(int(cells_per_group)):
                rows.append(
                    {
                        context_axis: context_value,
                        "perturbation": perturbation,
                    }
                )
    return pd.DataFrame(rows)


def _count_by_context_and_perturbation(
    obs: pd.DataFrame,
    indices: np.ndarray,
    context_axis: str,
) -> dict[tuple[object, object], int]:
    subset = obs.iloc[np.asarray(indices, dtype=np.int64)]
    grouped = subset.groupby([context_axis, "perturbation"], sort=True, observed=False).size()
    return {key: int(value) for key, value in grouped.to_dict().items()}


class ContextSplitterTests(unittest.TestCase):
    def test_in_context_prunes_test_to_holdout_context_and_held_out_perturbations(
        self,
    ) -> None:
        obs = _make_obs(
            context_axis="cell_line",
            context_values=["A", "B"],
            perturbations=["control", "gene1", "gene2", "gene3"],
        )
        splitter = ContextSplitter(
            adata=AnnDataProxy(obs),
            split_strategy="in-context",
            test_context_values=["A"],
            control_label="control",
        )

        _, _, test_idx = splitter.split(seed=0)
        metadata = splitter.get_split_metadata()
        test_obs = obs.iloc[test_idx]
        held_out_obs = test_obs[test_obs["cell_line"] == "A"]

        self.assertEqual(metadata.context_axis, "cell_line")
        self.assertIn("A", set(test_obs["cell_line"]))
        self.assertSetEqual(
            set(held_out_obs["perturbation"]),
            {"control", *metadata.held_out_perturbations},
        )

    def test_cross_context_uses_same_held_out_perturbations_for_multiple_contexts(
        self,
    ) -> None:
        obs = _make_obs(
            context_axis="donor",
            context_values=["D1", "D2", "D3"],
            perturbations=["control", "gene1", "gene2", "gene3", "gene4"],
        )
        splitter = ContextSplitter(
            adata=AnnDataProxy(obs),
            split_strategy="cross-context",
            context_axis="donor",
            test_context_values=["D2", "D3"],
            control_label="control",
        )

        _, _, test_idx = splitter.split(seed=3)
        metadata = splitter.get_split_metadata()
        test_obs = obs.iloc[test_idx]
        held_out = set(metadata.held_out_perturbations)

        self.assertEqual(metadata.context_axis, "donor")
        self.assertEqual(metadata.test_context_values, ("D2", "D3"))
        self.assertEqual(len(held_out), 2)
        self.assertSetEqual(set(test_obs["donor"]), {"D2", "D3"})

        for donor in ("D2", "D3"):
            donor_non_control = set(
                test_obs.loc[
                    (test_obs["donor"] == donor) & (test_obs["perturbation"] != "control"),
                    "perturbation",
                ]
            )
            self.assertSetEqual(donor_non_control, held_out)

    def test_cross_context_test_counts_match_in_context_targets_after_alignment(
        self,
    ) -> None:
        obs = _make_obs(
            context_axis="cell_line",
            context_values=["A", "B", "C"],
            perturbations=["control", "gene1", "gene2", "gene3", "gene4"],
        )
        in_splitter = ContextSplitter(
            adata=AnnDataProxy(obs),
            split_strategy="in-context",
            test_context_values=["B", "C"],
            control_label="control",
        )
        cross_splitter = ContextSplitter(
            adata=AnnDataProxy(obs),
            split_strategy="cross-context",
            test_context_values=["B", "C"],
            control_label="control",
        )

        in_train, _, in_test = in_splitter.split(seed=7)
        cross_train, _, cross_test = cross_splitter.split(seed=7)

        in_metadata = in_splitter.get_split_metadata()
        cross_metadata = cross_splitter.get_split_metadata()
        in_counts = _count_by_context_and_perturbation(obs, in_test, context_axis="cell_line")
        cross_counts = _count_by_context_and_perturbation(obs, cross_test, context_axis="cell_line")

        self.assertEqual(in_metadata.held_out_perturbations, cross_metadata.held_out_perturbations)
        self.assertEqual(len(in_train), len(cross_train))

        for context_value in in_metadata.test_context_values:
            for perturbation in in_metadata.held_out_perturbations:
                key = (context_value, perturbation)
                self.assertEqual(in_counts.get(key, 0), cross_counts.get(key, 0))

    def test_cross_context_test_cells_match_in_context_targets_after_alignment(
        self,
    ) -> None:
        obs = _make_obs(
            context_axis="cell_line",
            context_values=["A", "B", "C"],
            perturbations=["control", "gene1", "gene2", "gene3", "gene4"],
        )
        in_splitter = ContextSplitter(
            adata=AnnDataProxy(obs),
            split_strategy="in-context",
            test_context_values=["B", "C"],
            control_label="control",
        )
        cross_splitter = ContextSplitter(
            adata=AnnDataProxy(obs),
            split_strategy="cross-context",
            test_context_values=["B", "C"],
            control_label="control",
        )

        _, _, in_test = in_splitter.split(seed=7)
        _, _, cross_test = cross_splitter.split(seed=7)
        in_metadata = in_splitter.get_split_metadata()
        cross_metadata = cross_splitter.get_split_metadata()

        self.assertEqual(in_metadata.held_out_perturbations, cross_metadata.held_out_perturbations)

        in_test_set = set(np.asarray(in_test, dtype=np.int64).tolist())
        cross_test_set = set(np.asarray(cross_test, dtype=np.int64).tolist())

        for context_value in in_metadata.test_context_values:
            for perturbation in in_metadata.held_out_perturbations:
                expected = set(
                    obs.index[
                        (obs["cell_line"] == context_value) & (obs["perturbation"] == perturbation)
                    ].tolist()
                )
                self.assertSetEqual(in_test_set & expected, cross_test_set & expected)

    def test_single_context_dataset_defaults_to_only_context_value(self) -> None:
        obs = _make_obs(
            context_axis="cell_line",
            context_values=["solo"],
            perturbations=["control", "gene1", "gene2"],
        )
        splitter = ContextSplitter(
            adata=AnnDataProxy(obs),
            split_strategy="in-context",
            control_label="control",
        )

        splitter.split(seed=1)
        metadata = splitter.get_split_metadata()
        eval_contexts = build_evaluation_contexts(splitter)

        self.assertEqual(metadata.test_context_values, ("solo",))
        self.assertEqual(metadata.held_out_perturbations, ())
        self.assertEqual(len(eval_contexts), 1)
        self.assertEqual(eval_contexts[0].values, ("solo",))

    def test_cross_context_requires_multiple_context_values(self) -> None:
        obs = _make_obs(
            context_axis="cell_line",
            context_values=["solo"],
            perturbations=["control", "gene1", "gene2"],
        )

        with self.assertRaisesRegex(ValueError, "at least two distinct context values"):
            ContextSplitter(
                adata=AnnDataProxy(obs),
                split_strategy="cross-context",
                control_label="control",
            )

    def test_context_axis_defaults_to_cell_line(self) -> None:
        both_obs = pd.DataFrame(
            {
                "cell_line": ["A", "A"],
                "donor": ["D1", "D1"],
                "perturbation": ["control", "gene1"],
            }
        )
        neither_obs = pd.DataFrame(
            {
                "batch": ["x", "y"],
                "perturbation": ["control", "gene1"],
            }
        )

        splitter = ContextSplitter(
            adata=AnnDataProxy(both_obs),
            split_strategy="in-context",
            control_label="control",
            test_context_values=["A"],
        )

        self.assertEqual(splitter.context_axis, "cell_line")

        with self.assertRaisesRegex(KeyError, "Context axis 'cell_line'"):
            ContextSplitter(
                adata=AnnDataProxy(neither_obs),
                split_strategy="in-context",
                control_label="control",
                test_context_values=["x"],
            )


if __name__ == "__main__":
    unittest.main()
