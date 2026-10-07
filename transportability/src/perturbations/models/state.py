"""Wrapper utilities to train and run STATE on repository AnnData splits."""

from __future__ import annotations

import argparse
import importlib
import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy import sparse

_STATE_BATCH_KEY = "__state_batch__"
_STATE_BATCH_VALUE = "batch0"
_STATE_CELL_TYPE_CANDIDATES = ("cell_type", "cell_line", "donor")
_STATE_BATCH_CANDIDATES = (
    "batch",
    "batch_var",
    "gem_group",
    "donor",
    "plate",
    "experiment",
    "lane",
)


def _state_configs_root() -> Path:
    state_module = _load_state_module("state")
    module_path = getattr(state_module, "__file__", None)
    if not isinstance(module_path, str):
        raise RuntimeError("The installed 'state' package does not expose a module path.")
    return Path(module_path).resolve().parent / "configs"


def _load_state_module(module_name: str):
    """Load an upstream STATE module only when STATE execution is requested."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "state" or module_name.startswith(f"{exc.name}."):
            raise ModuleNotFoundError(
                "Running the STATE model requires the optional 'arc-state' package."
            ) from exc
        raise


def _load_state_entrypoint(module_name: str, function_name: str) -> Callable[[Any], None]:
    module = _load_state_module(module_name)
    return cast(Callable[[Any], None], getattr(module, function_name))


def _load_state_component_config(component_dir: str, component_name: str) -> DictConfig:
    config_path = _state_configs_root() / component_dir / f"{component_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected State config '{component_dir}/{component_name}.yaml' at {config_path}."
        )
    return OmegaConf.load(config_path)


@contextmanager
def _temporary_env(**updates: str | None):
    previous = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _disable_torch_weights_only_load():
    with _temporary_env(
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        TORCH_FORCE_WEIGHTS_ONLY_LOAD=None,
    ):
        yield


def _resolve_obs_key(
    obs: pd.DataFrame,
    preferred_key: str | None,
    *,
    candidates: tuple[str, ...],
    label: str,
    fallback: str | None = None,
) -> str:
    for candidate in (preferred_key, *candidates):
        if candidate is not None and candidate in obs.columns:
            return candidate
    if fallback is not None:
        return fallback
    raise KeyError(
        f"Could not determine a State {label} column. Available obs columns: {list(obs.columns)}"
    )


def _resolve_cell_type_key(obs: pd.DataFrame, preferred_key: str | None) -> str:
    return _resolve_obs_key(
        obs,
        preferred_key,
        candidates=_STATE_CELL_TYPE_CANDIDATES,
        label="cell-type/context",
    )


def _resolve_batch_key(obs: pd.DataFrame, preferred_key: str | None) -> str:
    return _resolve_obs_key(
        obs,
        preferred_key,
        candidates=_STATE_BATCH_CANDIDATES,
        label="batch",
        fallback=_STATE_BATCH_KEY,
    )


def _get_expression_matrix(adata: ad.AnnData, expression_layer: str | None):
    if expression_layer is None:
        return adata.X
    if expression_layer not in adata.layers:
        raise KeyError(
            f"Expression layer '{expression_layer}' not found. Available layers: {list(adata.layers.keys())}"
        )
    return adata.layers[expression_layer]


def _materialize_expression(matrix):
    if sparse.issparse(matrix):
        return matrix.astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _as_string_categories(values: pd.Series) -> pd.Categorical:
    return pd.Categorical(values.astype(str))


def _prepare_state_var(var: pd.DataFrame) -> pd.DataFrame:
    prepared_var = var.copy()
    gene_names = pd.Index(prepared_var.index.astype(str), name=prepared_var.index.name)
    prepared_var.index = gene_names

    for key in ("gene_name", "feature"):
        if key not in prepared_var.columns:
            prepared_var[key] = gene_names
    return prepared_var


def _prepare_state_obs(
    obs: pd.DataFrame,
    *,
    perturbation_key: str,
    cell_type_key: str,
    batch_key: str,
    control_label: str,
) -> pd.DataFrame:
    prepared_obs = obs.copy()
    if batch_key == _STATE_BATCH_KEY:
        prepared_obs[batch_key] = _STATE_BATCH_VALUE

    for key in (perturbation_key, cell_type_key, batch_key):
        prepared_obs[key] = _as_string_categories(prepared_obs[key])

    perturbations = prepared_obs[perturbation_key].astype(str)
    if control_label not in set(perturbations.unique()):
        raise ValueError(f"Control label '{control_label}' not found in obs['{perturbation_key}'].")
    return prepared_obs


def _prepare_state_adata(
    adata: ad.AnnData,
    *,
    expression_layer: str | None,
    cell_type_key: str | None,
    batch_key: str | None,
    perturbation_key: str,
    control_label: str,
) -> tuple[ad.AnnData, str, str]:
    if perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"Perturbation column '{perturbation_key}' not found. Available obs columns: {list(adata.obs.columns)}"
        )

    resolved_cell_type_key = _resolve_cell_type_key(adata.obs, cell_type_key)
    resolved_batch_key = _resolve_batch_key(adata.obs, batch_key)

    prepared = ad.AnnData(
        X=_materialize_expression(_get_expression_matrix(adata, expression_layer)),
        obs=_prepare_state_obs(
            adata.obs,
            perturbation_key=perturbation_key,
            cell_type_key=resolved_cell_type_key,
            batch_key=resolved_batch_key,
            control_label=control_label,
        ),
        var=_prepare_state_var(adata.var),
    )
    prepared.obs_names = adata.obs_names.copy()
    prepared.var_names = adata.var_names.copy()
    return prepared, resolved_cell_type_key, resolved_batch_key


def _combine_train_and_val(
    *,
    train_prepared: ad.AnnData,
    val_adata: ad.AnnData | None,
    expression_layer: str | None,
    perturbation_key: str,
    cell_type_key: str,
    batch_key: str,
    control_label: str,
) -> ad.AnnData:
    if val_adata is None or int(val_adata.n_obs) == 0:
        return train_prepared

    val_prepared, _, _ = _prepare_state_adata(
        val_adata,
        expression_layer=expression_layer,
        cell_type_key=cell_type_key,
        batch_key=batch_key,
        perturbation_key=perturbation_key,
        control_label=control_label,
    )
    return ad.concat([train_prepared, val_prepared], join="inner", merge="same")


def _write_state_toml(
    dataset_path: Path, toml_path: Path, dataset_name: str = "state_data"
) -> None:
    toml_path.write_text(
        "\n".join(
            [
                "[datasets]",
                f'{dataset_name} = "{dataset_path}"',
                "",
                "[training]",
                f'{dataset_name} = "train"',
                "",
                "[zeroshot]",
                "",
                "[fewshot]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_state_config(
    *,
    toml_path: Path,
    output_dir: Path,
    run_name: str,
    model_name: str,
    batch_size: int,
    max_steps: int,
    val_freq: int,
    seed: int,
    perturbation_key: str,
    cell_type_key: str,
    batch_key: str,
    control_label: str,
) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "name": run_name,
            "output_dir": str(output_dir),
            "use_wandb": False,
            "overwrite": True,
            "return_adatas": False,
            "pred_adata_path": None,
            "true_adata_path": None,
            "wandb": {
                "entity": "",
                "project": "state",
                "local_wandb_dir": str(output_dir / "wandb_logs"),
                "tags": [],
            },
        }
    )
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({"data": _load_state_component_config("data", "perturbation")}),
        OmegaConf.create({"model": _load_state_component_config("model", model_name)}),
        OmegaConf.create({"training": _load_state_component_config("training", "default")}),
    )

    cfg.data.kwargs.toml_config_path = str(toml_path)
    cfg.data.kwargs.embed_key = None
    cfg.data.kwargs.pert_col = perturbation_key
    cfg.data.kwargs.cell_type_key = cell_type_key
    cfg.data.kwargs.batch_col = batch_key
    cfg.data.kwargs.control_pert = control_label
    cfg.data.kwargs.output_space = "all"
    cfg.data.kwargs.num_workers = 0
    cfg.data.kwargs.pin_memory = False

    cfg.training.batch_size = int(batch_size)
    cfg.training.max_steps = int(max_steps)
    cfg.training.val_freq = max(1, min(int(max_steps), int(val_freq)))
    cfg.training.train_seed = int(seed)
    cfg.training.devices = 1
    cfg.training.use_mfu = False
    cfg.training.pop("cumulative_flops_use_backward", None)
    return cfg


def _build_infer_args(
    *,
    checkpoint_path: Path,
    adata_path: Path,
    output_path: Path,
    model_dir: Path,
    perturbation_key: str,
    cell_type_key: str,
    batch_key: str,
    control_label: str,
    seed: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(checkpoint_path),
        adata=str(adata_path),
        embed_key=None,
        pert_col=perturbation_key,
        output=str(output_path),
        model_dir=str(model_dir),
        celltype_col=cell_type_key,
        celltypes=None,
        batch_col=batch_key,
        control_pert=control_label,
        seed=int(seed),
        max_set_len=None,
        quiet=True,
        tsv=None,
        all_perts=False,
        virtual_cells_per_pert=None,
        min_cells=None,
        max_cells=None,
    )


def _finalize_state_prediction(
    *,
    state_prediction: ad.AnnData,
    test_adata: ad.AnnData,
    result_layer_key: str | None,
) -> ad.AnnData:
    result = test_adata[state_prediction.obs_names, state_prediction.var_names].copy()
    predicted_matrix = state_prediction.X.copy()

    if result_layer_key is None:
        result.X = predicted_matrix
    else:
        result.layers[result_layer_key] = predicted_matrix
    return result


def run_state(
    train_adata: ad.AnnData,
    val_adata: ad.AnnData | None,
    test_adata: ad.AnnData,
    *,
    expression_layer: str | None = None,
    result_layer_key: str | None = None,
    perturbation_key: str = "perturbation",
    cell_type_key: str | None = None,
    batch_key: str | None = None,
    control_label: str = "control",
    run_name: str = "state_run",
    model_name: str = "state_sm",
    batch_size: int = 16,
    max_steps: int = 200,
    val_freq: int = 200,
    seed: int = 42,
) -> ad.AnnData:
    """
    Train State on temporary AnnData files and return in-memory predictions.

    State's public interface expects perturbation/context group splits in a TOML file.
    This wrapper keeps the repository's existing cell-level splits unchanged by writing
    temporary train/test files and folding validation cells into the training set.
    """
    run_tx_train = _load_state_entrypoint("state._cli._tx._train", "run_tx_train")
    run_tx_infer = _load_state_entrypoint("state._cli._tx._infer", "run_tx_infer")

    train_prepared, resolved_cell_type_key, resolved_batch_key = _prepare_state_adata(
        train_adata,
        expression_layer=expression_layer,
        cell_type_key=cell_type_key,
        batch_key=batch_key,
        perturbation_key=perturbation_key,
        control_label=control_label,
    )
    train_source = _combine_train_and_val(
        train_prepared=train_prepared,
        val_adata=val_adata,
        expression_layer=expression_layer,
        perturbation_key=perturbation_key,
        cell_type_key=resolved_cell_type_key,
        batch_key=resolved_batch_key,
        control_label=control_label,
    )
    test_prepared, _, _ = _prepare_state_adata(
        test_adata,
        expression_layer=expression_layer,
        cell_type_key=resolved_cell_type_key,
        batch_key=resolved_batch_key,
        perturbation_key=perturbation_key,
        control_label=control_label,
    )

    with tempfile.TemporaryDirectory(prefix="state_run_") as work_dir_str:
        work_dir = Path(work_dir_str)
        os.environ.setdefault("MPLCONFIGDIR", str(work_dir / "mplconfig"))

        train_h5ad = work_dir / "train.h5ad"
        test_h5ad = work_dir / "test.h5ad"
        toml_path = work_dir / "state.toml"
        artifact_dir = work_dir / "artifacts"
        pred_out = work_dir / "test_predicted.h5ad"

        train_source.write_h5ad(train_h5ad)
        test_prepared.write_h5ad(test_h5ad)
        _write_state_toml(dataset_path=train_h5ad, toml_path=toml_path)

        run_tx_train(
            _build_state_config(
                toml_path=toml_path,
                output_dir=artifact_dir,
                run_name=run_name,
                model_name=model_name,
                batch_size=batch_size,
                max_steps=max_steps,
                val_freq=val_freq,
                seed=seed,
                perturbation_key=perturbation_key,
                cell_type_key=resolved_cell_type_key,
                batch_key=resolved_batch_key,
                control_label=control_label,
            )
        )

        run_dir = artifact_dir / run_name
        infer_args = _build_infer_args(
            checkpoint_path=run_dir / "checkpoints" / "final.ckpt",
            adata_path=test_h5ad,
            output_path=pred_out,
            model_dir=run_dir,
            perturbation_key=perturbation_key,
            cell_type_key=resolved_cell_type_key,
            batch_key=resolved_batch_key,
            control_label=control_label,
            seed=seed,
        )
        with _disable_torch_weights_only_load():
            run_tx_infer(infer_args)

        state_prediction = ad.read_h5ad(pred_out)

    return _finalize_state_prediction(
        state_prediction=state_prediction,
        test_adata=test_adata,
        result_layer_key=result_layer_key,
    )
