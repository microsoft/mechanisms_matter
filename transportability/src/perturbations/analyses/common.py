"""Shared analysis constants and helper utilities."""

from __future__ import annotations

MODELS = (
    "Control",
    "Average",
    "Context-Average",
    "Context-linearPCA",
    "linearPCA",
    "scVI",
    "GEARS",
    "CPA",
    "STATE",
    "scLDM",
)

NORM_LAYER_KEY = "normalized_log1p"


def label_to_target_tokens(label: str) -> tuple[str, ...]:
    """Split a perturbation label into non-control target tokens."""
    return tuple(token for token in str(label).split("+") if token and token != "control")
