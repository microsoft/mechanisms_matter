"""Fetch Geneformer model + dictionary assets for the extraction stage.

Downloads ONLY the files needed (one model directory + the pickle dictionaries)
from the ``ctheodoris/Geneformer`` Hugging Face repo, avoiding the multi-GB full
git-LFS checkout. Prints the resolved ``GENEFORMER_MODEL_DIR`` to export.

Run inside the isolated Geneformer env:

    python fetch_geneformer_assets.py --model Geneformer-V2-104M --dest ./gf_assets

Then:

    export GENEFORMER_MODEL_DIR="$(pwd)/gf_assets/Geneformer-V2-104M"

On AzureML, prefer pre-staging these files to your blob container once and mounting
them as a data asset; this script is the "download at runtime" alternative.

NOTE: model directory and dictionary file names differ across Geneformer releases.
Verify the names against https://huggingface.co/ctheodoris/Geneformer/tree/main
and adjust ``--model`` / ``--dict-patterns`` accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ID = "ctheodoris/Geneformer"
_DEFAULT_MODEL = "Geneformer-V2-104M"
# Dictionaries live at the repo root; pull all pickles (they are small).
_DEFAULT_DICT_PATTERNS = ("*.pkl", "geneformer/*.pkl")


def fetch_assets(
    model: str,
    dest: Path,
    dict_patterns: tuple[str, ...],
    revision: str | None,
) -> Path:
    """Download the chosen model dir + dictionary pickles into ``dest``."""
    from huggingface_hub import snapshot_download  # lazy import

    allow_patterns = [f"{model}/*", *dict_patterns]
    local_dir = snapshot_download(
        repo_id=_REPO_ID,
        allow_patterns=allow_patterns,
        local_dir=str(dest),
        revision=revision,
    )
    model_dir = Path(local_dir) / model
    return model_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL, help="Model subdirectory in the HF repo."
    )
    parser.add_argument("--dest", default="./gf_assets", help="Local download directory.")
    parser.add_argument(
        "--dict-patterns",
        nargs="*",
        default=list(_DEFAULT_DICT_PATTERNS),
        help="Glob patterns for dictionary files to include.",
    )
    parser.add_argument("--revision", default=None, help="Optional git revision/tag/commit.")
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    model_dir = fetch_assets(
        model=args.model,
        dest=dest,
        dict_patterns=tuple(args.dict_patterns),
        revision=args.revision,
    )
    print(f"Downloaded assets to: {dest}")
    print(f"GENEFORMER_MODEL_DIR={model_dir}")


if __name__ == "__main__":
    main()
