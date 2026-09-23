"""Check sweep sharding and durable results without generating data or training models."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from perturbations.analyses.synthetic_simulations import random_sweep as sweep


class RandomSweepShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(sweep, "MODELS", ("scLDM",)))
        self.enterContext(patch.dict(sweep._GLOBAL, {}, clear=True))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))

    def _run(self, output_dir: Path, *, n_trials: int, trial_start: int = 0) -> pd.DataFrame:
        return sweep.run_random_sweep(
            dataset_name="causalDGP",
            n_trials=n_trials,
            output_dir=output_dir,
            diversity_type="both",
            control_mu=np.ones(128),
            all_theta=np.ones(128),
            pert_mu=np.ones(128),
            gene_names=np.asarray([f"gene{i}" for i in range(128)]),
            rng=np.random.default_rng(42),
            use_multiprocessing=False,
            split_strategy="cross-context",
            trial_start=trial_start,
        )

    def test_shards_preserve_full_sweep_parameters_and_simulation_seeds(self) -> None:
        def fake_simulation(**kwargs):
            return [{"model": "scLDM", "simulation_seed": kwargs["trial_id_for_rng"]}]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with patch.object(sweep, "simulate_one_run", side_effect=fake_simulation):
                full = self._run(root / "full", n_trials=6)
                first = self._run(root / "first", n_trials=2)
                second = self._run(root / "second", n_trials=4, trial_start=2)

            combined = pd.concat([first, second], ignore_index=True)
            pd.testing.assert_frame_equal(
                full.sort_values("trial_id").reset_index(drop=True),
                combined.sort_values("trial_id").reset_index(drop=True),
                check_exact=True,
            )
            self.assertEqual(sorted(combined["trial_id"].tolist()), list(range(6)))
            self.assertEqual(combined["trial_id"].nunique(), 6)
            self.assertTrue((combined["simulation_seed"] == combined["trial_id"]).all())
            # The comparison exercises random parameter draws, not only fixed dimensions.
            self.assertGreater(full["B"].nunique(), 1)

    def test_completed_trial_csv_survives_interruption_during_next_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with patch.object(
                sweep,
                "simulate_one_run",
                side_effect=[
                    [{"model": "scLDM", "pearson": 0.75}],
                    KeyboardInterrupt("interrupted during the second trial"),
                ],
            ) as simulation:
                with self.assertRaises(KeyboardInterrupt):
                    self._run(output_dir, n_trials=3, trial_start=20)

            # Execution is ordered by estimated cost, independently of global trial IDs.
            self.assertEqual(simulation.call_count, 2)
            completed_trial_id = simulation.call_args_list[0].kwargs["trial_id_for_rng"]
            self.assertIn(completed_trial_id, range(20, 23))
            result_files = list(output_dir.glob("results_*.csv"))
            self.assertEqual(len(result_files), 1)
            saved = pd.read_csv(result_files[0])
            self.assertEqual(saved["trial_id"].tolist(), [completed_trial_id])
            self.assertEqual(saved["status"].tolist(), ["success"])
            self.assertEqual(saved["pearson"].tolist(), [0.75])
            self.assertEqual(saved["dataset"].tolist(), ["causalDGP"])
            self.assertEqual(saved["split_strategy"].tolist(), ["cross-context"])
            self.assertEqual(saved["diversity_type"].tolist(), ["both"])
            self.assertEqual(list(output_dir.glob("*.tmp")), [])

    def test_selected_models_reach_workers_and_failure_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.object(sweep, "MODELS", ("Control", "scLDM")):
                with patch.object(
                    sweep, "simulate_one_run", side_effect=ValueError("failed")
                ) as run:
                    result = sweep.run_random_sweep(
                        dataset_name="directDGP",
                        n_trials=1,
                        output_dir=temporary_dir,
                        models=("scLDM",),
                        control_mu=np.ones(128),
                        all_theta=np.ones(128),
                        pert_mu=np.ones(128),
                        gene_names=np.asarray([f"gene{i}" for i in range(128)]),
                        rng=np.random.default_rng(42),
                        use_multiprocessing=False,
                    )
        self.assertEqual(run.call_args.kwargs["models"], ("scLDM",))
        self.assertEqual(result["model"].tolist(), ["scLDM"])
        self.assertEqual(result["context_values"].tolist(), ["0"])
        self.assertEqual(result["status"].tolist(), ["failed"])

    def test_failed_worker_rows_preserve_trial_and_experiment_metadata(self) -> None:
        sweep.init_worker(
            control_mu=np.ones(128),
            all_theta=np.ones(128),
            pert_mu=np.ones(128),
            gene_names=np.asarray([f"gene{i}" for i in range(128)]),
        )
        params = sweep.sample_parameters(sweep.PARAM_RANGES, np.random.default_rng(42))
        task = {
            "dataset_name": "causalDGP",
            "trial_id": 73,
            "params_dict": params,
            "split_strategy": "in-context",
            "diversity_type": "b",
            "pid": 123,
        }
        errors = io.StringIO()
        with redirect_stderr(errors):
            with patch.object(
                sweep, "simulate_one_run", side_effect=ValueError("synthetic failure")
            ) as simulation:
                rows = sweep._pool_worker_timed(task)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key, value in params.items():
            self.assertEqual(row[key], value)
        self.assertEqual(row["trial_id"], 73)
        self.assertEqual(row["model"], "scLDM")
        self.assertEqual(row["dataset"], "causalDGP")
        self.assertEqual(row["split_strategy"], "in-context")
        self.assertEqual(row["diversity_type"], "b")
        self.assertEqual(row["context_axis"], "cell_line")
        self.assertEqual(row["context_values"], "1")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"], "synthetic failure")
        self.assertEqual(simulation.call_args.kwargs["trial_id_for_rng"], 73)
        self.assertIn("Traceback (most recent call last)", errors.getvalue())
        self.assertIn("ValueError: synthetic failure", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
