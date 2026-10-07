"""Regression checks for the optional STATE backend used by the CD4 benchmark."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

state = pytest.importorskip("perturbations.models.state_gene.state_gene")
loader = pytest.importorskip("perturbations.models.state_gene.loader")


def _adata(contexts: tuple[str, ...] = ("A", "B")) -> ad.AnnData:
    labels = [label for _ in contexts for label in ("control",) * 4 + ("target",) * 4]
    obs = pd.DataFrame(
        {
            "perturbation": labels,
            "context": [context for context in contexts for _ in range(8)],
        },
        index=[f"cell_{context}_{i}" for context in contexts for i in range(8)],
    )
    expression = np.random.default_rng(15).uniform(0.1, 2, (len(obs), 3)).astype(np.float32)
    result = ad.AnnData(expression, obs=obs)
    result.obsm["X_geneformer"] = expression[:, :2].copy()
    return result


def test_context_ids_are_shared_between_disjoint_splits() -> None:
    mapping = {"A": 0, "B": 1, "C": 2}
    train = loader.create_perturbation_dataloader(
        _adata(("A", "B")),
        "perturbation",
        "control",
        2,
        ["control", "target"],
        context_key="context",
        basal_embedding_key="X_geneformer",
        context_to_idx=mapping,
    )
    valid = loader.create_perturbation_dataloader(
        _adata(("B", "C")),
        "perturbation",
        "control",
        2,
        ["control", "target"],
        context_key="context",
        basal_embedding_key="X_geneformer",
        context_to_idx=mapping,
        pin_memory=True,
    )
    assert {sample[2] for sample in train.dataset.samples} == {0, 1}
    assert {sample[2] for sample in valid.dataset.samples} == {1, 2}
    assert valid.pin_memory
    sample = valid.dataset[len(valid.dataset) - 1]
    assert sample["batch"].unique().tolist() == [2]
    assert sample["ctrl_cell_emb"].shape == (2, 2)
    assert sample["pert_cell_emb"].shape == sample["ctrl_cell_gene"].shape == (2, 3)


def test_shared_context_ids_require_all_labels() -> None:
    with pytest.raises(ValueError, match="missing labels"):
        loader.PerturbationSetDataset(
            _adata(("B",)),
            "perturbation",
            "control",
            2,
            ["control", "target"],
            context_key="context",
            context_to_idx={"A": 0},
        )


def test_state_resumes_optimizer_epoch_callbacks_and_rejects_stale_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_trainer = state.pl.Trainer
    interrupt = True

    class StopAfterFirstEpoch(state.pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch == 1:
                raise RuntimeError("simulated interruption")

    def make_trainer(**kwargs):
        if interrupt:
            kwargs["callbacks"].append(StopAfterFirstEpoch())
        kwargs.update(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        kwargs["callbacks"] = [
            callback
            for callback in kwargs["callbacks"]
            if not isinstance(callback, state.LearningRateMonitor)
        ]
        return real_trainer(**kwargs)

    monkeypatch.setattr(state.pl, "Trainer", make_trainer)
    data = _adata()
    options = dict(
        train_adata=data,
        valid_adata=data.copy(),
        test_adata=data.copy(),
        context_key="context",
        basal_embedding_key="X_geneformer",
        seed=7,
        model_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        resume_from_checkpoint="last",
        hidden_dim=8,
        epochs=3,
        batch_size=1,
        device="cpu",
        dataloader_num_workers=0,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        state.run_state_gene(**options)
    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    interrupted = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert not interrupted["state_training_complete"]
    assert interrupted["epoch"] == 0
    assert interrupted["global_step"] == 2
    assert interrupted["optimizer_states"][0]["state"]
    assert any("EarlyStopping" in key for key in interrupted["callbacks"])

    interrupt = False
    output = state.run_state_gene(**options)
    assert output["global_step"] == 6
    assert output["epochs_completed"] == 3
    assert output["input_dim"] == 2
    assert output["output_dim"] == 3
    assert output["preds"].shape == (data.n_obs, 3)
    assert np.isfinite(output["preds"]).all()
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert final["state_training_complete"]
    optimizer_steps = [
        int(value["step"].item()) for value in final["optimizer_states"][0]["state"].values()
    ]
    assert set(optimizer_steps) == {6}
    assert any("EarlyStopping" in key for key in final["callbacks"])

    # A retry after failed downstream evaluation uses the completed checkpoint.
    retry = state.run_state_gene(**options)
    assert retry["global_step"] == output["global_step"]
    np.testing.assert_array_equal(retry["preds"], output["preds"])

    with pytest.raises(ValueError, match="provenance"):
        state.run_state_gene(**(options | {"seed": 8}))
    changed = data.copy()
    changed.obsm["X_geneformer"][0, 0] += 0.1
    with pytest.raises(ValueError, match="provenance"):
        state.run_state_gene(**(options | {"train_adata": changed}))


@pytest.mark.parametrize("invalid", ["missing", "width", "nan"])
def test_state_rejects_invalid_embeddings_before_training(invalid: str, tmp_path: Path) -> None:
    data = _adata()
    valid = data.copy()
    if invalid == "missing":
        del valid.obsm["X_geneformer"]
    elif invalid == "width":
        valid.obsm["X_geneformer"] = np.zeros((valid.n_obs, 4), dtype=np.float32)
    else:
        valid.obsm["X_geneformer"][0, 0] = np.nan
    with pytest.raises((KeyError, ValueError), match=r"basal|Basal"):
        state.run_state_gene(
            data,
            valid,
            data.copy(),
            "context",
            basal_embedding_key="X_geneformer",
            model_dir=str(tmp_path),
            epochs=1,
        )
