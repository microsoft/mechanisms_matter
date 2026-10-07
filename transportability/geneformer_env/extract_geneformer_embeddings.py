r"""
Extract Geneformer cell embeddings and store them in ``adata.obsm``.

This is the encoder-extraction half of the "foundation-model encoder" ablation:
we run a pretrained Geneformer model over every cell to obtain a fixed-width
cell embedding (the ``<cls>`` token representation), which is then consumed by
``run_state_gene`` as the basal state via its ``basal_embedding_key`` argument.

Pipeline
--------
1. Map ``adata.var_names`` (gene symbols) to Ensembl IDs (Geneformer's vocabulary
   key). An existing Ensembl column in ``.var`` is used when available; the
   fraction of matched genes is reported.
2. Prepare a tokenizer-ready AnnData (raw counts in ``.X``, ``obs['n_counts']``,
   and a stable ``obs['cell_index']`` used to realign embeddings afterwards).
3. Tokenize with Geneformer's ``TranscriptomeTokenizer`` (rank-value encoding).
4. Extract per-cell embeddings with ``EmbExtractor`` (``emb_mode='cls'``).
5. Reorder embeddings to match the input AnnData row order and write them to
   ``adata.obsm[obsm_key]`` (default ``X_geneformer``).

Geneformer is an optional heavy dependency; it is imported lazily so that
importing this module (e.g. for tests) does not require it. This script lives in
``geneformer_env`` (not the STATE/transportability package) because it must run
in the isolated Geneformer environment, which pins an older ``transformers`` that
is incompatible with the STATE env. It has no intra-package imports, so run it by
path from that env.

Because the documented setup uses ``GIT_LFS_SKIP_SMUDGE=1``, the dictionaries bundled
inside the installed ``geneformer`` package are unfetched git-LFS pointer files. The
real ones are downloaded next to the model by ``fetch_geneformer_assets.py``, so this
script locates them and passes them to the tokenizer/extractor explicitly rather than
letting Geneformer fall back to the stubs.

Usage
-----
    cd transportability/geneformer_env
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    uv run python extract_geneformer_embeddings.py \\
        --input ../src/perturbations/data/norman19/norman19_processed.h5ad \\
        --output ../src/perturbations/data/norman19/norman19_geneformer.h5ad \\
        --counts-layer counts \\
        --model-dir "$GENEFORMER_MODEL_DIR" \\
        --obsm-key X_geneformer

The tokenizer dictionaries and the gene-symbol mapping are auto-discovered from the
asset directory holding ``--model-dir``; use ``--dictionary-dir`` to point elsewhere.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

_CELL_INDEX_COL = "cell_index"
_ENSEMBL_COL = "ensembl_id"
_DEFAULT_OBSM_KEY = "X_geneformer"

# Glob patterns for the tokenizer dictionaries, keyed by the Geneformer kwarg they
# feed. Names are release-specific (e.g. ``*_gc104M.pkl`` for V2), hence the globs.
_TOKENIZER_DICT_PATTERNS = {
    "gene_median_file": "gene_median_dictionary*.pkl",
    "token_dictionary_file": "token_dictionary*.pkl",
    "gene_mapping_file": "ensembl_mapping_dict*.pkl",
}
_SYMBOL_DICT_PATTERN = "gene_name_id_dict*.pkl"

# Curated legacy-symbol -> current-HGNC-symbol renames, applied as a pre-pass
# before the gene_name_id_dict lookup so datasets carrying older symbols still map.
# An alias is only used if the renamed symbol actually exists in the authoritative
# gene_name_id_dict, so a wrong entry here can never invent a false Ensembl ID.
_DEFAULT_HGNC_ALIASES: dict[str, str] = {
    # Cytoplasmic aminoacyl-tRNA synthetases gained a "1" suffix (HGNC ~2021).
    "AARS": "AARS1",
    "CARS": "CARS1",
    "DARS": "DARS1",
    "EPRS": "EPRS1",
    "GARS": "GARS1",
    "HARS": "HARS1",
    "IARS": "IARS1",
    "KARS": "KARS1",
    "LARS": "LARS1",
    "MARS": "MARS1",
    "NARS": "NARS1",
    "QARS": "QARS1",
    "RARS": "RARS1",
    "SARS": "SARS1",
    "TARS": "TARS1",
    "VARS": "VARS1",
    "WARS": "WARS1",
    "YARS": "YARS1",
    # Other well-documented renames.
    "ADPRHL2": "ADPRS",
    "H3F3A": "H3-3A",
    "H3F3B": "H3-3B",
    "HIST2H2AC": "H2AC20",
}


def _looks_like_ensembl(values: pd.Series[Any]) -> bool:
    """Return True when the majority of values look like Ensembl gene IDs."""
    stripped = values.astype(str).str.split(".").str[0]
    return bool(stripped.str.startswith("ENSG").mean() > 0.5)


def _assign_ensembl(adata: ad.AnnData, series: pd.Series[Any]) -> float:
    """Write a normalized ``ensembl_id`` column and return the matched fraction."""
    ensembl = pd.Series(series).astype(str).str.split(".").str[0]
    valid = ensembl.str.startswith("ENSG") & (ensembl != "nan")
    adata.var[_ENSEMBL_COL] = np.where(valid.to_numpy(), ensembl.to_numpy(), "")
    return float(valid.mean())


def _load_symbol_to_ensembl(path: str) -> dict[str, str]:
    """Load a pickle mapping gene symbol -> Ensembl ID (e.g. gene_name_id_dict)."""
    import pickle

    with Path(path).open("rb") as handle:
        mapping = pickle.load(handle)
    return {str(k): str(v) for k, v in dict(mapping).items()}


def _load_alias_map(path: str) -> dict[str, str]:
    """Load an old-symbol -> new-symbol alias map from JSON or a 2-column CSV."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        import json

        with file_path.open() as handle:
            return {str(k): str(v) for k, v in json.load(handle).items()}
    frame = pd.read_csv(file_path, header=None)
    return {
        str(old): str(new) for old, new in zip(frame.iloc[:, 0], frame.iloc[:, 1], strict=False)
    }


def _resolve_ensembl_ids(
    adata: ad.AnnData,
    ensembl_col: str | None,
    gene_name_id_dict: str | None = None,
    alias_map: str | None = None,
) -> float:
    """
    Populate ``adata.var['ensembl_id']`` and return the matched fraction.

    Resolution order: explicit ``ensembl_col`` -> ``var_names`` that already look
    like Ensembl IDs -> common Ensembl columns -> gene-symbol mapping via
    ``gene_name_id_dict`` (a pickle mapping gene symbol -> Ensembl ID, such as
    Geneformer's bundled ``gene_name_id_dict_gc104M.pkl``). This handles datasets
    like Replogle22/Norman19 whose ``var_names`` are gene symbols with no Ensembl
    column.
    """
    # 1) Explicit column.
    if ensembl_col and ensembl_col in adata.var.columns:
        return _assign_ensembl(adata, adata.var[ensembl_col])

    # 2) var_names are already Ensembl IDs.
    var_names = pd.Series(adata.var_names.astype(str), index=adata.var_names)
    if _looks_like_ensembl(var_names):
        return _assign_ensembl(adata, var_names)

    # 3) Common Ensembl columns.
    for col in ("ensembl_id", "gene_ids", "gene_id", "ensembl", "ensembl_ids"):
        if col in adata.var.columns and _looks_like_ensembl(adata.var[col]):
            return _assign_ensembl(adata, adata.var[col])

    # 4) Map gene symbols -> Ensembl via a provided dictionary, with an
    #    HGNC-alias pre-pass for legacy/renamed symbols.
    if gene_name_id_dict:
        mapping = _load_symbol_to_ensembl(gene_name_id_dict)
        aliases = dict(_DEFAULT_HGNC_ALIASES)
        if alias_map:
            aliases.update(_load_alias_map(alias_map))

        symbols = pd.Series(adata.var_names.astype(str), index=adata.var_names)
        recovered = 0

        def _to_ensembl(symbol: str) -> str:
            nonlocal recovered
            if symbol in mapping:
                return mapping[symbol]
            renamed = aliases.get(symbol)
            if renamed and renamed in mapping:
                recovered += 1
                return mapping[renamed]
            return ""

        mapped = symbols.map(_to_ensembl)
        if recovered:
            print(f"Recovered {recovered} renamed HGNC symbol(s) via alias pre-pass.")
        return _assign_ensembl(adata, mapped)

    raise KeyError(
        "Could not resolve Ensembl IDs. adata.var_names are gene symbols and no "
        "Ensembl column was found. Pass --ensembl-col (a var column with Ensembl "
        "IDs), or --gene-name-id-dict pointing at a pickle mapping gene symbol -> "
        "Ensembl ID (e.g. Geneformer's gene_name_id_dict_gc104M.pkl, fetched by "
        f"fetch_geneformer_assets.py). Available var columns: {list(adata.var.columns)}"
    )


def _is_lfs_pointer(path: Path) -> bool:
    """Return True when ``path`` is an unfetched git-LFS pointer rather than real data."""
    try:
        with path.open("rb") as handle:
            return handle.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _dictionary_candidates(model_dir: str, dictionary_dir: str | None) -> list[Path]:
    """Directories to search for Geneformer dictionaries, most-specific first."""
    model_path = Path(model_dir)
    candidates: list[Path] = []
    if dictionary_dir:
        candidates.append(Path(dictionary_dir))
    # fetch_geneformer_assets.py lays assets out as <dest>/<model>/ and <dest>/geneformer/.
    candidates += [model_path.parent / "geneformer", model_path, model_path.parent]
    # Packaged copies last: usable only when the repo was cloned with git-LFS smudging.
    try:
        import geneformer  # type: ignore

        if geneformer.__file__:
            candidates.append(Path(geneformer.__file__).parent)
    except Exception:  # package layout is best-effort; missing copies are fine
        pass
    return [directory for directory in candidates if directory.is_dir()]


def _find_dictionary(candidates: list[Path], pattern: str) -> Path | None:
    """Return the first real (non-LFS-stub) file matching ``pattern`` in ``candidates``."""
    for directory in candidates:
        for match in sorted(directory.glob(pattern)):
            if match.is_file() and not _is_lfs_pointer(match):
                return match
    return None


def _resolve_tokenizer_dictionaries(candidates: list[Path]) -> dict[str, str]:
    """
    Locate the tokenizer dictionaries Geneformer needs, as real files.

    Geneformer defaults these to copies bundled inside the installed package, but the
    documented ``GIT_LFS_SKIP_SMUDGE=1 uv sync`` leaves those as ~130-byte git-LFS
    pointers, which surface later as a cryptic ``UnpicklingError: invalid load key,
    'v'``. Resolving them up front lets us fail with actionable guidance instead.
    """
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for kwarg, pattern in _TOKENIZER_DICT_PATTERNS.items():
        found = _find_dictionary(candidates, pattern)
        if found is None:
            missing.append(pattern)
        else:
            resolved[kwarg] = str(found)

    if missing:
        searched = "\n  ".join(str(directory) for directory in candidates) or "(none)"
        raise FileNotFoundError(
            "Could not find usable Geneformer tokenizer dictionaries for: "
            f"{', '.join(missing)}.\nSearched:\n  {searched}\n"
            "The dictionaries bundled with the pip/git install are git-LFS pointer "
            "stubs when installed with GIT_LFS_SKIP_SMUDGE=1. Download the real ones "
            "with fetch_geneformer_assets.py (they land in <dest>/geneformer/), then "
            "pass --dictionary-dir if they are not next to --model-dir."
        )
    return resolved


def _install_cpu_device_shim() -> bool:
    """
    Redirect Geneformer's hardcoded CUDA tensor placements to CPU.

    Geneformer loads the *model* behind a ``torch.cuda.is_available()`` check but then
    forces its *input* tensors onto ``"cuda"`` unconditionally (``emb_extractor`` and
    ``perturber_utils``). On the CPU torch build this project installs by default, that
    mismatch raises "Torch not compiled with CUDA enabled". Remapping the placements is
    less invasive than vendoring a patched Geneformer. No-op when a GPU is present, so
    GPU runs are unaffected. Returns True when the shim was installed.
    """
    import torch

    if torch.cuda.is_available():
        return False

    def _is_cuda(value: Any) -> bool:
        if isinstance(value, torch.device):
            return value.type == "cuda"
        return isinstance(value, str) and value.startswith("cuda")

    real_tensor = torch.tensor
    real_to = torch.Tensor.to

    def _tensor(*args: Any, **kwargs: Any) -> Any:
        if _is_cuda(kwargs.get("device")):
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    def _to(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args and _is_cuda(args[0]):
            args = ("cpu", *args[1:])
        elif _is_cuda(kwargs.get("device")):
            kwargs["device"] = "cpu"
        return real_to(self, *args, **kwargs)

    torch.tensor = _tensor  # type: ignore[assignment]
    torch.Tensor.to = _to  # type: ignore[assignment,method-assign]
    torch.cuda.empty_cache = lambda: None  # type: ignore[assignment]
    return True


def _prepare_tokenizer_adata(
    adata: ad.AnnData,
    counts_layer: str | None,
) -> ad.AnnData:
    """Build a tokenizer-ready AnnData with raw counts, n_counts, and cell_index."""
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"counts_layer='{counts_layer}' not found. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        counts: Any = adata.layers[counts_layer]
    else:
        counts = adata.X

    counts = counts.copy()
    prepared = ad.AnnData(X=counts, var=adata.var.copy())
    prepared.var[_ENSEMBL_COL] = adata.var[_ENSEMBL_COL].to_numpy()

    library_size = counts.sum(axis=1)
    library_size = np.asarray(library_size).reshape(-1)
    prepared.obs["n_counts"] = library_size.astype(np.float64)
    # Stable index so we can realign embeddings to the original row order.
    prepared.obs[_CELL_INDEX_COL] = np.arange(adata.n_obs, dtype=np.int64)
    return prepared


def extract_geneformer_embeddings(
    adata: ad.AnnData,
    model_dir: str,
    counts_layer: str | None = "counts",
    ensembl_col: str | None = None,
    gene_name_id_dict: str | None = None,
    alias_map: str | None = None,
    dictionary_dir: str | None = None,
    obsm_key: str = _DEFAULT_OBSM_KEY,
    emb_mode: str = "cls",
    model_input_size: int = 4096,
    special_token: bool = True,
    nproc: int = 4,
    max_ncells: int | None = None,
    forward_batch_size: int = 64,
) -> ad.AnnData:
    """
    Compute Geneformer cell embeddings and attach them to ``adata.obsm``.

    Returns the same ``adata`` with ``obsm[obsm_key]`` populated (shape
    ``(n_cells, emb_dim)``). Rows are aligned to the input AnnData order.
    """
    # Lazy import: Geneformer is a heavy, optional dependency.
    try:
        from geneformer import EmbExtractor, TranscriptomeTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "Geneformer is not installed. Install it (e.g. `uv add geneformer` or from "
            "https://huggingface.co/ctheodoris/Geneformer) to extract embeddings."
        ) from exc

    if _install_cpu_device_shim():
        print("No CUDA device found; running Geneformer on CPU.")

    candidates = _dictionary_candidates(model_dir, dictionary_dir)
    tokenizer_dicts = _resolve_tokenizer_dictionaries(candidates)

    # var_names are often gene symbols (Norman19/Replogle22); auto-discover the
    # symbol -> Ensembl dictionary alongside the other assets when not supplied.
    if not gene_name_id_dict:
        discovered = _find_dictionary(candidates, _SYMBOL_DICT_PATTERN)
        if discovered is not None:
            gene_name_id_dict = str(discovered)

    matched_fraction = _resolve_ensembl_ids(
        adata, ensembl_col, gene_name_id_dict=gene_name_id_dict, alias_map=alias_map
    )
    print(
        f"Ensembl ID coverage: {matched_fraction:.1%} of {adata.n_vars} genes mapped to ENSG IDs."
    )
    if matched_fraction == 0.0:
        raise ValueError(
            "No genes mapped to Ensembl IDs; check --ensembl-col / --gene-name-id-dict and adata.var."
        )

    prepared = _prepare_tokenizer_adata(adata, counts_layer=counts_layer)

    with tempfile.TemporaryDirectory() as tmp_root:
        data_dir = Path(tmp_root) / "input"
        token_dir = Path(tmp_root) / "tokenized"
        emb_dir = Path(tmp_root) / "embeddings"
        data_dir.mkdir(parents=True, exist_ok=True)
        token_dir.mkdir(parents=True, exist_ok=True)
        emb_dir.mkdir(parents=True, exist_ok=True)

        input_h5ad = data_dir / "cells.h5ad"
        prepared.write_h5ad(input_h5ad)

        # Carry cell_index through tokenization so we can realign afterwards.
        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={_CELL_INDEX_COL: _CELL_INDEX_COL},
            nproc=nproc,
            model_input_size=model_input_size,
            special_token=special_token,
            **tokenizer_dicts,
        )
        tokenizer.tokenize_data(
            data_directory=str(data_dir),
            output_directory=str(token_dir),
            output_prefix="cells",
            file_format="h5ad",
        )
        tokenized_dataset = token_dir / "cells.dataset"

        emb_extractor = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode=emb_mode,
            max_ncells=max_ncells,
            emb_layer=-1,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
            emb_label=[_CELL_INDEX_COL],
            token_dictionary_file=tokenizer_dicts["token_dictionary_file"],
        )
        emb_df: pd.DataFrame = emb_extractor.extract_embs(
            model_directory=model_dir,
            input_data_file=str(tokenized_dataset),
            output_directory=str(emb_dir),
            output_prefix="cells_emb",
        )

    embeddings = _embeddings_from_dataframe(emb_df, n_cells=adata.n_obs)
    adata.obsm[obsm_key] = embeddings
    print(
        f"Stored Geneformer embeddings in adata.obsm['{obsm_key}'] with shape {embeddings.shape}."
    )
    return adata


def _embeddings_from_dataframe(emb_df: pd.DataFrame, n_cells: int) -> np.ndarray:
    """Realign an EmbExtractor DataFrame to the original row order."""
    if _CELL_INDEX_COL not in emb_df.columns:
        raise KeyError(
            f"Expected '{_CELL_INDEX_COL}' column in the embedding output; "
            f"got columns: {list(emb_df.columns)[:10]}..."
        )
    emb_cols = [c for c in emb_df.columns if c != _CELL_INDEX_COL]
    ordered = emb_df.sort_values(_CELL_INDEX_COL)
    cell_ids = ordered[_CELL_INDEX_COL].to_numpy().astype(np.int64)
    if cell_ids.shape[0] != n_cells or not np.array_equal(cell_ids, np.arange(n_cells)):
        raise ValueError(
            "Embedding rows do not align one-to-one with input cells "
            f"(got {cell_ids.shape[0]} rows for {n_cells} cells). Some cells may "
            "have been dropped during tokenization (e.g. too few detected genes)."
        )
    return np.asarray(ordered[emb_cols].to_numpy(), dtype=np.float32)


def _none_if_empty(value: str | None) -> str | None:
    """Return ``None`` for empty-string CLI values."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the input .h5ad.")
    parser.add_argument("--output", required=True, help="Path to write the .h5ad with embeddings.")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("GENEFORMER_MODEL_DIR"),
        help="Pretrained Geneformer model directory (or set GENEFORMER_MODEL_DIR).",
    )
    parser.add_argument(
        "--counts-layer",
        default="counts",
        help="Layer with raw counts (use '' to read from .X).",
    )
    parser.add_argument(
        "--ensembl-col",
        default=None,
        help="var column holding Ensembl IDs (autodetected when omitted).",
    )
    parser.add_argument(
        "--gene-name-id-dict",
        default=None,
        help=(
            "Pickle mapping gene symbol -> Ensembl ID, used when var_names are gene "
            "symbols and no Ensembl column exists (e.g. Geneformer's "
            "gene_name_id_dict_gc104M.pkl). Auto-discovered from the asset directory "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--dictionary-dir",
        default=os.environ.get("GENEFORMER_DICTIONARY_DIR"),
        help=(
            "Directory holding Geneformer's *.pkl dictionaries (gene_median, "
            "token_dictionary, ensembl_mapping_dict). Defaults to searching alongside "
            "--model-dir, which is where fetch_geneformer_assets.py puts them."
        ),
    )
    parser.add_argument(
        "--alias-map",
        default=None,
        help=(
            "Optional JSON or 2-column CSV mapping legacy gene symbol -> current HGNC "
            "symbol, applied before the gene_name_id_dict lookup to recover renamed "
            "symbols. Extends the built-in default alias table."
        ),
    )
    parser.add_argument("--obsm-key", default=_DEFAULT_OBSM_KEY)
    parser.add_argument("--emb-mode", default="cls", choices=["cls", "cell"])
    parser.add_argument("--model-input-size", type=int, default=4096)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--forward-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-ncells",
        type=int,
        default=None,
        help="Optional cap on number of cells embedded (debugging).",
    )
    return parser


def main() -> None:
    """CLI entry point for Geneformer embedding extraction."""
    args = _build_arg_parser().parse_args()
    if not args.model_dir:
        raise SystemExit(
            "No Geneformer model directory provided. Pass --model-dir or set GENEFORMER_MODEL_DIR."
        )

    adata = ad.read_h5ad(args.input)
    extract_geneformer_embeddings(
        adata,
        model_dir=args.model_dir,
        counts_layer=_none_if_empty(args.counts_layer),
        ensembl_col=args.ensembl_col,
        gene_name_id_dict=_none_if_empty(args.gene_name_id_dict),
        alias_map=_none_if_empty(args.alias_map),
        dictionary_dir=_none_if_empty(args.dictionary_dir),
        obsm_key=args.obsm_key,
        emb_mode=args.emb_mode,
        model_input_size=args.model_input_size,
        nproc=args.nproc,
        max_ncells=args.max_ncells,
        forward_batch_size=args.forward_batch_size,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
