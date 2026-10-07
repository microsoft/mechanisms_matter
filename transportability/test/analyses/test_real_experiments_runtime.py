"""CPU-only coverage for safe scLDM checkpoint reuse in both real-data routes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from perturbations.analyses.real_experiments import run


@pytest.fixture(params=("h5ad", "cd4"))
def trial(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Exercise the actual route closures with evaluation and training mocked out."""
    adata = ad.AnnData(
        X=np.zeros((6, 2), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(6)]),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    indices = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    metadata = SimpleNamespace(context_axis="context")
    splitter = SimpleNamespace(
        split_strategy="in-context",
        obs=adata.obs.copy(),
        split=lambda seed: tuple(indices),
        get_split_metadata=lambda: metadata,
    )
    prepared = SimpleNamespace(test_obs=adata.obs.iloc[4:].copy(), test_var=adata.var.copy())
    runtime = SimpleNamespace(var_names=adata.var_names, obsm_widths={})
    backend = Mock(return_value={"preds": np.ones((2, 2), dtype=np.float32)})
    monkeypatch.setenv("SCLDM_OUTPUT_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(run, "run_scldm", backend)
    monkeypatch.setattr(run, "_prepare_trial_data", lambda **kwargs: prepared)
    monkeypatch.setattr(run, "_prepare_cd4_trial_data_from_cache", lambda **kwargs: prepared)
    monkeypatch.setattr(run, "_materialize_cd4_split_cache", lambda **kwargs: {})
    monkeypatch.setattr(
        run,
        "_run_model_from_cd4_split_cache",
        lambda *, split_paths, run_model: run_model(*(adata[index] for index in indices)),
    )
    monkeypatch.setattr(run, "_run_trial_models", lambda **kwargs: kwargs["run_scldm"]())
    options = {
        "splitter": splitter,
        "trial_id": 7,
        "dataset_run_name": "fixture",
        "counts_layer": "counts",
        "obs_layer": "normalized_log1p",
        "pid": 123,
        "norm_target_sum": 1e4,
        "models": ("scLDM",),
    }

    def invoke():
        if request.param == "h5ad":
            return run.run_one_trial(adata=adata, **options)
        return run.run_one_trial_cd4_chunked(
            runtime=runtime, split_cache_parent=tmp_path, **options
        )

    output = tmp_path / "models" / "trial_7"
    return SimpleNamespace(
        invoke=invoke,
        output=output,
        marker=output / "scldm_resume_identity.json",
        backend=backend,
        adata=adata,
        runtime=runtime,
        splitter=splitter,
        metadata=metadata,
        indices=indices,
        options=options,
    )


def test_matching_retry_preserves_resume_and_output_path(trial: SimpleNamespace) -> None:
    trial.invoke()
    original = trial.marker.read_bytes()
    checkpoint = trial.output / "vae_ckpts" / "last.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not loaded by the identity guard")
    trial.invoke()
    assert trial.backend.call_count == 2
    assert trial.marker.read_bytes() == original
    assert trial.backend.call_args.kwargs["out_dir"] == str(trial.output)
    assert trial.backend.call_args.kwargs["resume"] is True
    assert trial.backend.call_args.kwargs["seed"] == 7
    assert json.loads(original)["split_strategy"] == "in-context"
    assert sorted(path.name for path in trial.output.iterdir()) == [
        "scldm_resume_identity.json",
        "vae_ckpts",
    ]


@pytest.mark.parametrize(
    "change",
    (
        "strategy",
        "split_membership",
        "split_order",
        "obs_order",
        "gene_order",
        "dataset",
        "context",
        "counts",
        "normalization",
    ),
)
def test_incompatible_retry_never_calls_backend(trial: SimpleNamespace, change: str) -> None:
    trial.invoke()
    original = trial.marker.read_bytes()
    trial.backend.reset_mock()
    if change == "strategy":
        trial.splitter.split_strategy = "cross-context"
    elif change == "split_membership":
        trial.indices[0], trial.indices[1] = trial.indices[1], trial.indices[0]
    elif change == "split_order":
        trial.indices[0] = trial.indices[0][::-1]
    elif change == "obs_order":
        trial.splitter.obs = trial.splitter.obs.iloc[::-1]
    elif change == "gene_order":
        trial.adata.var_names = trial.adata.var_names[::-1]
        trial.runtime.var_names = trial.adata.var_names
    elif change == "dataset":
        trial.options["dataset_run_name"] = "different_dataset"
    elif change == "context":
        trial.metadata.context_axis = "different_context"
    elif change == "counts":
        trial.options["counts_layer"] = "different_counts"
    else:
        trial.options["norm_target_sum"] = 2e4
    with pytest.raises(ValueError, match="resume identity mismatch"):
        trial.invoke()
    trial.backend.assert_not_called()
    assert trial.marker.read_bytes() == original


@pytest.mark.parametrize(
    "checkpoint_name", ("vae_ckpts/last.ckpt", "ldm_ckpts/last.ckpt", "vae.ckpt", "ldm.ckpt")
)
def test_legacy_checkpoint_requires_identity(trial: SimpleNamespace, checkpoint_name: str) -> None:
    checkpoint = trial.output / checkpoint_name
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"must not be read")
    with pytest.raises(ValueError, match="no resume identity marker"):
        trial.invoke()
    trial.backend.assert_not_called()
    assert not trial.marker.exists()


@pytest.mark.parametrize("payload", ("{", "[]", "{}", "null"))
def test_malformed_marker_never_calls_backend(trial: SimpleNamespace, payload: str) -> None:
    trial.output.mkdir(parents=True)
    trial.marker.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="resume identity"):
        trial.invoke()
    trial.backend.assert_not_called()
    assert trial.marker.read_text(encoding="utf-8") == payload


def test_competing_identity_is_not_replaced(
    trial: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a competing publisher appearing after the initial existence check."""
    payload = json.dumps({"version": 1, "dataset": "competing_dataset"})

    def competing_link(source, destination):
        destination.write_text(payload, encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(run.os, "link", competing_link)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        trial.invoke()
    trial.backend.assert_not_called()
    assert trial.marker.read_text(encoding="utf-8") == payload
    assert list(trial.output.iterdir()) == [trial.marker]
