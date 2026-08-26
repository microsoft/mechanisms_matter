"""One-shot import/health check for the Geneformer extraction environment.

Run inside the isolated Geneformer env (NOT the STATE env):

    python check_env.py

Exits non-zero if Geneformer or its dependencies cannot be imported. This is the
fastest way to confirm the transformers-version conflict is resolved before
launching an AzureML job.
"""

from __future__ import annotations

import importlib
import sys

_REQUIRED = ("geneformer", "transformers", "torch", "anndata", "datasets", "huggingface_hub")


def main() -> int:
    """Import required modules and the Geneformer entry points; print versions."""
    ok = True
    for name in _REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            ok = False
            print(f"{name}: FAILED -> {exc}")

    # The symbol that breaks under new transformers lives behind these imports.
    try:
        from geneformer import EmbExtractor, TranscriptomeTokenizer  # noqa: F401

        print("geneformer TranscriptomeTokenizer + EmbExtractor import OK")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"geneformer entry points: FAILED -> {exc}")

    print("ENV OK" if ok else "ENV BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
