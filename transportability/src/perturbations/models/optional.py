"""Utilities for loading optional model implementations on demand."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast


def lazy_model_runner(
    module_name: str,
    function_name: str,
    model_name: str,
) -> Callable[..., Any]:
    """Return a callable that imports an optional model only when invoked."""

    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"{model_name} is unavailable because its optional dependencies are not installed."
            ) from error

        implementation = cast(Callable[..., Any], getattr(module, function_name))
        return implementation(*args, **kwargs)

    return run
