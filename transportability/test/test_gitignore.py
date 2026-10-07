import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
IGNORED_MODEL_PATHS = (
    "transportability/src/perturbations/models/cpa/cpa.py",
    "transportability/src/perturbations/models/gears/gears.py",
    "transportability/src/perturbations/models/scldm/models.py",
    "transportability/src/perturbations/models/state_gene/state_gene.py",
    "transportability/src/perturbations/models/state.py",
)


@pytest.mark.parametrize("model_path", IGNORED_MODEL_PATHS)
def test_third_party_model_is_gitignored(model_path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "--", model_path],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert result.returncode == 0, f"Expected {model_path} to be ignored by Git"
