import pytest

from perturbations.models.optional import lazy_model_runner


def test_lazy_model_runner_defers_missing_module_error_until_called() -> None:
    runner = lazy_model_runner(
        "perturbations.models.not_in_open_source_release",
        "run_model",
        "ExampleModel",
    )

    with pytest.raises(ModuleNotFoundError, match="ExampleModel is unavailable"):
        runner()


def test_lazy_model_runner_calls_available_implementation() -> None:
    runner = lazy_model_runner("math", "sqrt", "SquareRoot")

    assert runner(9.0) == 3.0
