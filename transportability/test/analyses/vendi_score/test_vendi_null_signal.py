from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from perturbations.analyses.vendi_score import vendi_null_signal


def test_null_signal_sweep_pairs_null_and_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        vendi_null_signal,
        "load_parameter_estimation_inputs",
        lambda: {"unused": np.empty(0)},
    )

    def fake_generate_and_score(**kwargs: Any) -> dict[str, object]:
        calls.append(kwargs)
        factor = float(kwargs["effect_factor"])
        return {
            "P": int(kwargs["P"]),
            "G": int(kwargs["G"]),
            "N0": int(kwargs["N0"]),
            "Nk": int(kwargs["Nk"]),
            "p_effect": float(kwargs["p_effect"]),
            "effect_factor": factor,
            "seed": int(kwargs["seed"]),
            "vendi_score_pseudobulk": factor,
            "effective_rank_pseudobulk": factor,
        }

    monkeypatch.setattr(
        vendi_null_signal,
        "_generate_and_score_pseudobulk",
        fake_generate_and_score,
    )

    results = vendi_null_signal.run_null_signal_sweep(
        repeats=2,
        P=4,
        G=8,
        N0=16,
        Nk_values=(4, 8),
        effect_factors=(1.0, 2.0),
    )

    assert len(calls) == 8
    assert len(results) == 8
    null_rows = results.loc[results["effect_factor"] == 1.0]
    signal_rows = results.loc[results["effect_factor"] == 2.0]
    assert null_rows["is_population_null"].all()
    assert (null_rows["expected_null_diversity"] == 1.0).all()
    assert not signal_rows["is_population_null"].any()
    assert signal_rows["expected_null_diversity"].isna().all()

    paired_counts = results.groupby(["Nk", "repeat", "seed"])["effect_factor"].nunique()
    assert (paired_counts == 2).all()


def test_null_signal_sweep_requires_matched_control_splits() -> None:
    with pytest.raises(ValueError, match="twice the largest Nk"):
        vendi_null_signal.run_null_signal_sweep(
            repeats=1,
            P=4,
            G=8,
            N0=15,
            Nk_values=(8,),
            effect_factors=(1.0, 2.0),
        )


def test_summarize_and_plot_null_signal_sweep(tmp_path: Path) -> None:
    rows: list[dict[str, float | int | bool]] = []
    for Nk in (4, 8):
        for seed in range(2):
            for effect_factor in (1.0, 2.0):
                rows.append(
                    {
                        "P": 4,
                        "Nk": Nk,
                        "p_effect": 0.05,
                        "effect_factor": effect_factor,
                        "seed": seed,
                        "is_population_null": effect_factor == 1.0,
                        "expected_null_diversity": (1.0 if effect_factor == 1.0 else float("nan")),
                        "vendi_score_pseudobulk": effect_factor + 0.1 * seed,
                        "effective_rank_pseudobulk": effect_factor + 0.2 * seed,
                    }
                )
    summary = vendi_null_signal.summarize_null_signal_sweep(pd.DataFrame(rows))

    vendi_null_signal.plot_null_signal_sweep(summary, tmp_path)

    assert len(summary) == 4
    output_path = tmp_path / "vendi_null_signal.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
