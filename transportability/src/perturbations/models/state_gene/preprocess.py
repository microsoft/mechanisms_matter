"""Preprocessing utilities for state-gene models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
import torch
from esm import pretrained
from tqdm import tqdm

if TYPE_CHECKING:
    import anndata

log = logging.getLogger(__name__)


def _fetch_protein_sequences(gene_names: list[str]) -> dict[str, str]:
    """Fetch reviewed human protein sequences from UniProt for each gene symbol."""
    gene_to_seq: dict[str, str] = {}
    for gene in tqdm(gene_names, desc="Fetching protein sequences from UniProt"):
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                    "format": "fasta",
                    "size": "1",
                },
                timeout=10,
            )
            if resp.status_code == 200 and resp.text.strip():
                lines = resp.text.strip().split("\n")
                seq = "".join(line for line in lines if not line.startswith(">"))
                if seq:
                    gene_to_seq[gene] = seq
        except Exception:
            continue
    return gene_to_seq


_ESM2_MODELS: dict[str, tuple[str, int]] = {
    "esm2_t33_650M_UR50D": ("esm2_t33_650M_UR50D", 33),  # 1280-dim, ~2.5GB
    "esm2_t48_15B_UR50D": ("esm2_t48_15B_UR50D", 48),  # 5120-dim, ~60GB
}


def generate_esm2_embeddings(
    gene_names: list[str],
    output_path: str | None = None,
    model_name: str = "esm2_t33_650M_UR50D",
) -> dict[str, torch.Tensor] | None:
    """
    Generate ESM-2 protein embeddings for a list of gene symbols.

    Fetches protein sequences from UniProt, runs them through an ESM-2 model,
    and returns mean-pooled last-layer representations.

    Args:
        gene_names: Gene symbols to embed (e.g. ["TP53", "BRCA1", ...]).
        output_path: If provided, save the embeddings dict to this .pt file.
        model_name: ESM-2 model to use. "esm2_t33_650M_UR50D" (1280-dim, ~2.5GB)
                    or "esm2_t48_15B_UR50D" (5120-dim, ~60GB).

    Returns:
        Dict mapping gene symbol (uppercased) to embedding tensor, or None if
        fewer than 50% of genes had sequences available.
    """
    if model_name not in _ESM2_MODELS:
        raise ValueError(f"Unknown model {model_name}. Choose from {list(_ESM2_MODELS)}")
    loader_name, repr_layer = _ESM2_MODELS[model_name]

    # 1) Fetch protein sequences
    log.info("Fetching protein sequences for %d genes...", len(gene_names))
    gene_sequences = _fetch_protein_sequences(gene_names)
    log.info("Found sequences for %d / %d genes", len(gene_sequences), len(gene_names))

    if len(gene_sequences) < len(gene_names) * 0.5:
        log.warning("Too few sequences found (<50%%), falling back to caller")
        return None

    # 2) Load ESM-2 model
    log.info("Loading %s...", model_name)
    loader_fn = getattr(pretrained, loader_name)
    esm_model, alphabet = loader_fn()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()

    # 3) Compute embeddings
    gene_to_embedding: dict[str, torch.Tensor] = {}
    for gene_name, protein_seq in tqdm(gene_sequences.items(), desc="Computing ESM-2 embeddings"):
        protein_seq = protein_seq[:1022]  # truncate to avoid OOM
        _, _, batch_tokens = batch_converter([(gene_name, protein_seq)])
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[repr_layer])
        # Mean-pool over sequence length (excluding BOS/EOS special tokens)
        embedding = results["representations"][repr_layer][0, 1:-1].mean(0).cpu()
        gene_to_embedding[gene_name.upper()] = embedding

    if output_path and gene_to_embedding:
        torch.save(gene_to_embedding, output_path)
        log.info("Saved ESM-2 embeddings to %s", output_path)

    return gene_to_embedding if gene_to_embedding else None


def create_onehot_embeddings(all_genes: set[str]) -> dict[str, torch.Tensor]:
    """Make one-hot embeddings for each gene."""
    genes_sorted = sorted(all_genes)
    emb: dict[str, torch.Tensor] = {}
    for i, g in enumerate(genes_sorted):
        vec = torch.zeros(len(genes_sorted))
        vec[i] = 1.0
        emb[g] = vec
    return emb


def preprocess_from_adata(
    adatas: dict[str, anndata.AnnData],
    output_dir: str,
    profile_name: str,
    all_embeddings_path: str | None = None,
    gene_column: str | None = None,
    try_esm2: bool = True,
    esm2_model: str = "esm2_t33_650M_UR50D",
) -> dict[str, Any]:
    """
    Build embedding artifacts directly from AnnData objects.

    Priority: load from file → generate ESM-2 → one-hot fallback.

    Args:
        adatas: {"train": train_adata, "val": valid_adata}
        output_dir: where to write artifacts
        profile_name: name for output files
        all_embeddings_path: path to ESM2 .pt file, or None to try generating
        gene_column: var column with gene names, or None to use var_names
        try_esm2: if True, attempt to generate ESM-2 embeddings before falling
                  back to one-hot
        esm2_model: which ESM-2 model to use (default: esm2_t33_650M_UR50D)

    Returns:
        dict with paths to all_embeddings, ds_emb_mapping, valid_genes_masks
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect all gene names across datasets
    all_genes: set[str] = set()
    for adata in adatas.values():
        genes = adata.var[gene_column] if gene_column else adata.var_names
        all_genes.update(str(g).upper() for g in genes)

    emb_file = str(out_path / f"all_embeddings_{profile_name}.pt")

    # Priority: provided file → generate ESM-2 → one-hot fallback
    all_embeddings: dict[str, torch.Tensor]
    if all_embeddings_path and Path(all_embeddings_path).exists():
        log.info("Loading existing embeddings from %s", all_embeddings_path)
        loaded = torch.load(all_embeddings_path)
        all_embeddings = {str(k).upper(): v for k, v in loaded.items()}
    elif try_esm2:
        log.info("Attempting to generate ESM-2 embeddings with %s...", esm2_model)
        esm_result = generate_esm2_embeddings(
            sorted(all_genes), output_path=emb_file, model_name=esm2_model
        )
        if esm_result is None:
            log.info("ESM-2 generation failed, falling back to one-hot embeddings")
            all_embeddings = create_onehot_embeddings(all_genes)
        else:
            all_embeddings = esm_result
    else:
        log.info("Using one-hot embeddings for %d genes", len(all_genes))
        all_embeddings = create_onehot_embeddings(all_genes)

    gene_to_idx = {g: i for i, g in enumerate(all_embeddings.keys())}

    # Save embeddings
    torch.save(all_embeddings, emb_file)

    # Build per-dataset mapping and masks
    ds_map: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for name, adata in adatas.items():
        genes = adata.var[gene_column] if gene_column else adata.var_names
        genes_upper = [str(g).upper() for g in genes]
        mapping = torch.tensor([gene_to_idx.get(g, -1) for g in genes_upper], dtype=torch.long)
        mask = mapping != -1
        ds_map[name] = mapping
        masks[name] = mask

    mapping_file = str(out_path / f"ds_emb_mapping_{profile_name}.torch")
    torch.save(ds_map, mapping_file)

    masks_file = str(out_path / f"valid_genes_masks_{profile_name}.torch")
    torch.save(masks, masks_file)

    return {
        "all_embeddings": emb_file,
        "ds_emb_mapping": mapping_file,
        "valid_genes_masks": masks_file,
        "size": next(iter(all_embeddings.values())).shape[0],
        "num": len(all_embeddings),
    }
