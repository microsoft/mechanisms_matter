# Adapted from https://github.com/ArcInstitute/state/tree/4fbfbecd1e6f0e2d151e06fdbf08549d9fbf3ecf/src/state
"""StateTransitionPerturbationModel: a set-based transformer model for perturbation prediction using optimal transport."""

import logging
from typing import Any, cast

import anndata
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.sparse import issparse  # type: ignore
from scipy.stats import pearsonr  # type: ignore
from transformers import PreTrainedModel

from ...util.anndata_util import materialize_adata
from ..test_synthentic_data import generate_synthetic_perturbation_data
from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .loader import create_perturbation_dataloader
from .utils import (
    apply_lora,
    build_mlp,
    get_activation_class,
    get_transformer_backbone,
)

logger = logging.getLogger(__name__)
_seed = 42
torch.manual_seed(_seed)  # type: ignore
np.random.seed(_seed)  # noqa
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)


class CombinedLoss(nn.Module):
    """Combined Sinkhorn + Energy loss."""

    def __init__(
        self, sinkhorn_weight: float = 0.001, energy_weight: float = 1.0, blur: float = 0.05
    ):
        """Initialize the combined loss with specified weights for each component."""
        super().__init__()  # type: ignore
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss as a weighted sum of Sinkhorn and Energy losses."""
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val


class ConfidenceToken(nn.Module):
    """Learnable confidence token that gets appended to the input sequence and learns to predict the expected loss value."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """Initialize the confidence token and its projection head."""
        super().__init__()  # type: ignore
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(
        self, transformer_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred


class StateTransitionPerturbationModel(PerturbationModel):
    """
    Set-based transformer model for perturbation prediction using optimal transport.

    1) Projects basal expression and perturbation encodings into a shared latent space.
    2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
    3) Enables cells to attend to one another, learning a set-to-set function rather than
    a sample-to-sample single-cell map.
    """

    gene_names: list[str] | None = None
    gene_decoder: nn.Module | None = None
    training: bool = True
    log: Any  # Lightning's log method; pyright can't resolve it from the MRO
    transformer_backbone: PreTrainedModel

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int,
        gene_dim: int,
        transformer_backbone_kwargs: dict[str, object],
        basal_mapping_strategy: str = "random",
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        output_space: str = "gene",
        **kwargs: Any,
    ):
        """
        Initialize the StateTransitionPerturbationModel.

        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
            batch_dim: number of batches (for batch encoding, if used)
            distributional_loss: which geomloss SamplesLoss to use ("energy", "sinkhorn", etc.)
            gene_dim: number of genes in the output space (if output_space is "gene")
            output_space: "gene" or "latent" - whether the model outputs directly to gene space or to a latent embedding space
            predict_residual: whether the model should predict a residual to the basal state (True) or the full perturbed state (False)
            transformer_backbone_key: which transformer architecture to use as the backbone (e.g. "GPT2")
            transformer_backbone_kwargs: kwargs to specify the transformer backbone architecture (e.g. number of layers, heads, etc.)
            basal_mapping_strategy: how to map basal state to latent space ("random" for random init, "shared" for shared encoder with perturbation, "separate" for separate encoder)
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get(
            "extra_tokens", 0
        )

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim
        self.mmd_num_chunks = max(int(kwargs.get("mmd_num_chunks", 1)), 1)
        self.randomize_mmd_chunks = bool(kwargs.get("randomize_mmd_chunks", False))

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(
                sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur
            )
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks(lora_cfg=kwargs.get("lora", None))

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False):
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # Optional batch predictor ablation: learns a single batch token added to every position,
        # and adds an auxiliary per-token batch classification head + CE loss.
        self.batch_predictor = bool(kwargs.get("batch_predictor", False))
        # If batch_encoder is enabled, disable batch_predictor per request
        if self.batch_encoder is not None and self.batch_predictor:
            logger.warning(
                "Both model.kwargs.batch_encoder and model.kwargs.batch_predictor are True. Disabling batch_predictor and proceeding with batch_encoder."
            )
            self.batch_predictor = False
            try:
                # Keep hparams in sync if available
                self.hparams["batch_predictor"] = False  # type: ignore[index]
            except Exception:
                pass

        self.batch_predictor_weight = float(kwargs.get("batch_predictor_weight", 0.1))
        self.batch_predictor_num_classes: int | None = batch_dim if self.batch_predictor else None
        if self.batch_predictor:
            if self.batch_predictor_num_classes is None:
                raise ValueError(
                    "batch_predictor=True requires a valid `batch_dim` (number of batch classes)."
                )
            # A single learnable batch token that is added to each position
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            # Simple per-token classifier from transformer hidden to batch classes
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_predictor_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=4,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None
        # Internal cache for last token features (B, S, H) from transformer for aux loss
        self._token_features: torch.Tensor | None = None

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if (
            is_gene_space or getattr(self, "gene_decoder", None) is None
        ):  # logically if we have a gene decoder, we are not directly outputting gene space, so don't apply relu to the main output
            self.relu = torch.nn.ReLU()

        self.use_batch_token = kwargs.get("use_batch_token", False)
        self.basal_mapping_strategy = basal_mapping_strategy
        # Disable batch token only for truly incompatible cases
        disable_reasons: list[str] = []
        if self.batch_encoder and self.use_batch_token:
            disable_reasons.append("batch encoder is used")

        if disable_reasons:
            self.use_batch_token = False
            logger.warning(
                f"Batch token is not supported when {' or '.join(disable_reasons)}, setting use_batch_token to False"
            )
            try:
                self.hparams["use_batch_token"] = False  ## type: ignore[index]
            except Exception:
                pass

        self.batch_token_weight = kwargs.get("batch_token_weight", 0.1)
        self.batch_token_num_classes: int | None = batch_dim if self.use_batch_token else None

        if self.use_batch_token:
            if self.batch_token_num_classes is None:
                raise ValueError("batch_token_num_classes must be set when use_batch_token is True")
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_token_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=1,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None

        # Internal cache for last token features (B, S, H) from transformer for aux loss
        self._batch_token_cache: torch.Tensor | None = None

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(
                hidden_dim=self.hidden_dim, dropout=self.dropout
            )
            self.confidence_loss_fn = nn.MSELoss()
            self.confidence_target_scale = float(kwargs.get("confidence_target_scale", 10.0))
            self.confidence_weight = float(kwargs.get("confidence_weight", 0.01))
        else:
            self.confidence_target_scale = None
            self.confidence_weight = 0.0

        # Backward-compat: accept legacy key `freeze_pert`
        self.freeze_pert_backbone = kwargs.get(
            "freeze_pert_backbone", kwargs.get("freeze_pert", False)
        )
        if self.freeze_pert_backbone:
            # Freeze backbone base weights but keep LoRA adapter weights (if present) trainable
            for name, param in self.transformer_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Freeze projection head as before
            for param in self.project_out.parameters():
                param.requires_grad = False

        # control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            # Prefer the gene names supplied by the data module (aligned to training output)
            gene_names = self.gene_names
            if gene_names is None:
                raise ValueError(
                    "finetune_vci_decoder=True but model.gene_names is None. Please provide gene_names via data module var_dims."
                )

            n_genes = len(gene_names)
            logger.info(
                f"Initializing FinetuneVCICountsDecoder with {n_genes} genes (output_space={output_space}; "
                + ("HVG subset" if output_space == "gene" else "all genes")
                + ")"
            )
            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
            )
        print(self)

    def _build_networks(self, lora_cfg: dict[str, Any] | None = None) -> None:
        """Here we instantiate the actual GPT2-based model."""
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Disable positional embeddings: cell sentences are unordered sets,
        # so positional encoding would leak spurious ordering information.
        if hasattr(self.transformer_backbone, "wpe"):
            self.transformer_backbone.wpe.weight.data.zero_()  # pyright: ignore
            self.transformer_backbone.wpe.weight.requires_grad = False  # pyright: ignore

        # Optionally wrap backbone with LoRA adapters
        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def forward(
        self, batch: dict[str, torch.Tensor], padded: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        The main forward call.

        Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        # Shape: [B, S, input_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        if self.use_batch_token and self.batch_token is not None:
            batch_size, _, _ = seq_input.shape
            # Prepend the batch token to the sequence along the sequence dimension
            # [B, S, H] -> [B, S+1, H], batch token at position 0
            seq_input = torch.cat([self.batch_token.expand(batch_size, -1, -1), seq_input], dim=1)

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E] (might be one more if we have the batch token)
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):  # # type: ignore[union-attr]
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"  # pyright: ignore

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(
                1, 1, seq_length, seq_length
            )

            # Get number of attention heads from model config
            num_heads = self.transformer_backbone.config.num_attention_heads

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        # Extract outputs accounting for optional prepended batch token and optional confidence token at the end
        if (
            self.confidence_token is not None
            and self.use_batch_token
            and self.batch_token is not None
        ):
            # transformer_output: [B, 1 + S + 1, H] -> batch token at 0, cells 1..S, confidence at -1
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(
                transformer_output[:, 1:, :]
            )
            # res_pred currently excludes the confidence token and starts from former index 1
            self._batch_token_cache = batch_token_pred
        elif self.confidence_token is not None:
            # Only confidence token appended at the end
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(
                transformer_output
            )
            self._batch_token_cache = None
        elif self.use_batch_token and self.batch_token is not None:
            # Only batch token prepended at the beginning
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred = transformer_output[:, 1:, :]  # [B, S, H]
            self._batch_token_cache = batch_token_pred
        else:
            # Neither special token used
            res_pred = transformer_output
            self._batch_token_cache = None

        # Cache token features for auxiliary batch prediction loss (B, S, H)
        self._token_features = res_pred

        # add to basal if predicting residual
        if self.predict_residual and self.output_space == "all":
            # Project control_cells to hidden_dim space to match res_pred
            # control_cells_hidden = self.project_to_hidden(control_cells)
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply relu if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None  # type: ignore[union-attr]
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if confidence_pred is not None:
            return output, confidence_pred
        else:
            return output

    def _compute_distribution_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply the primary distributional loss, optionally chunking feature dimensions for SamplesLoss."""
        if isinstance(self.loss_fn, SamplesLoss) and self.mmd_num_chunks > 1:
            feature_dim = pred.shape[-1]
            num_chunks = min(self.mmd_num_chunks, feature_dim)
            if num_chunks > 1 and feature_dim > 0:
                if self.randomize_mmd_chunks and self.training:
                    perm = torch.randperm(feature_dim, device=pred.device)
                    pred = pred.index_select(-1, perm)
                    target = target.index_select(-1, perm)
                pred_chunks = torch.chunk(pred, num_chunks, dim=-1)
                target_chunks = torch.chunk(target, num_chunks, dim=-1)
                chunk_losses = [
                    self.loss_fn(p_chunk, t_chunk)
                    for p_chunk, t_chunk in zip(pred_chunks, target_chunks, strict=True)
                ]
                return torch.stack(chunk_losses, dim=0).nanmean(dim=0)

        return self.loss_fn(pred, target)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, padded: bool = True
    ) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        target = batch["pert_cell_emb"]
        confidence_pred = None

        if self.confidence_token is not None:
            result = self.forward(batch, padded=padded)
            assert isinstance(result, tuple)
            pred, confidence_pred = result
        else:
            result = self.forward(batch, padded=padded)
            assert isinstance(result, torch.Tensor)
            pred = result

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        per_set_main_losses = self._compute_distribution_loss(pred, target)
        main_loss = torch.nanmean(per_set_main_losses)
        self.log("train_loss", main_loss)

        # Log individual loss components if using combined loss
        # if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
        #     sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).nanmean()
        #     energy_component = self.loss_fn.energy_loss(pred, target).nanmean()
        #     self.log("train/sinkhorn_loss", sinkhorn_component)
        #     self.log("train/energy_loss", energy_component)
        if isinstance(self.loss_fn, CombinedLoss):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).nanmean()
            energy_component = self.loss_fn.energy_loss(pred, target).nanmean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if (
            self.use_batch_token
            and self.batch_classifier is not None
            and self._batch_token_cache is not None
        ):
            logits = self.batch_classifier(self._batch_token_cache)  # [B, 1, C]
            batch_token_targets = batch["batch"]

            B = logits.shape[0]
            C = logits.size(-1)

            # Prepare one label per sequence (all S cells share the same batch)
            if batch_token_targets.dim() > 1 and batch_token_targets.size(-1) == C:
                # One-hot labels; reshape to [B, S, C]
                if padded:
                    target_oh = batch_token_targets.reshape(-1, self.cell_sentence_len, C)
                else:
                    target_oh = batch_token_targets.reshape(1, -1, C)
                sentence_batch_labels = target_oh.argmax(-1)
            else:
                # Integer labels; reshape to [B, S]
                if padded:
                    sentence_batch_labels = batch_token_targets.reshape(-1, self.cell_sentence_len)
                else:
                    sentence_batch_labels = batch_token_targets.reshape(1, -1)

            if sentence_batch_labels.shape[0] != B:
                sentence_batch_labels = sentence_batch_labels.reshape(B, -1)

            if self.basal_mapping_strategy == "batch":
                uniform_mask = sentence_batch_labels.eq(sentence_batch_labels[:, :1]).all(dim=1)
                if not torch.all(uniform_mask):
                    bad_indices = torch.where(~uniform_mask)[0]
                    label_strings: list[str] = []
                    for idx in bad_indices:
                        # labels = sentence_batch_labels[idx].detach().cpu().tolist()
                        labels: list[int] = sentence_batch_labels[idx].detach().cpu().tolist()  # type: ignore[assignment]
                        logger.error("Batch labels for sentence %d: %s", idx.item(), labels)
                        label_strings.append(f"sentence {idx.item()}: {labels}")
                    raise ValueError(
                        f"Expected all cells in a sentence to share the same batch when basal_mapping_strategy is 'batch'.\nFound mixed batch labels: {', '.join(label_strings)}"
                    )

            target_idx = sentence_batch_labels[:, 0]

            # Safety: ensure exactly one target per sequence
            if target_idx.numel() != B:
                target_idx = target_idx.reshape(-1)[:B]

            ce_loss = F.cross_entropy(logits.reshape(B, -1, C).squeeze(1), target_idx.long())
            self.log("train/batch_token_loss", ce_loss)
            total_loss = total_loss + self.batch_token_weight * ce_loss

        # Auxiliary batch prediction loss (per token), if enabled
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            assert isinstance(self.gene_decoder, FinetuneVCICountsDecoder)
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:  # noqa: NPY002
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            if padded:
                gene_targets = gene_targets.reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
            else:
                gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

            decoder_per_set = self._compute_distribution_loss(pert_cell_counts_preds, gene_targets)
            decoder_loss = decoder_per_set.mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            confidence_pred_vals = confidence_pred
            if confidence_pred_vals.dim() > 1:
                confidence_pred_vals = confidence_pred_vals.squeeze(-1)
            confidence_targets = per_set_main_losses.detach()
            if self.confidence_target_scale is not None:
                confidence_targets = confidence_targets * self.confidence_target_scale
            confidence_targets = confidence_targets.to(confidence_pred_vals.device)
            assert self.confidence_loss_fn is not None
            confidence_loss = self.confidence_weight * self.confidence_loss_fn(
                confidence_pred_vals, confidence_targets
            )
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", confidence_targets.mean())

            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            # Residual (delta-to-basal) L1 sparsity penalty, computed in gene
            # space. In the baseline (input_dim == output_dim) the basal input is
            # already gene expression. When the basal is a precomputed embedding
            # (input_dim != output_dim), use the control cells' gene expression
            # (carried as "ctrl_cell_gene") so the delta remains a valid
            # gene-space perturbation effect.
            basal_for_delta = (
                batch["ctrl_cell_gene"] if "ctrl_cell_gene" in batch else batch["ctrl_cell_emb"]
            )
            basal_for_delta = basal_for_delta.reshape_as(pred)
            delta = pred - basal_for_delta

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        return total_loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Validation step logic."""
        if self.confidence_token is None:
            pred = self.forward(batch)
            assert isinstance(pred, torch.Tensor)
            confidence_pred = None
        else:
            pred, confidence_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        per_set_main_losses = self._compute_distribution_loss(pred, target)
        loss = torch.nanmean(per_set_main_losses)
        self.log("val_loss", loss)

        # Log individual loss components if using combined loss
        # if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
        #     sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
        #     energy_component = self.loss_fn.energy_loss(pred, target).mean()
        if isinstance(self.loss_fn, CombinedLoss):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component)
            self.log("val/energy_loss", energy_component)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            assert isinstance(self.gene_decoder, FinetuneVCICountsDecoder)
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
            )
            gene_targets = gene_targets.reshape(
                -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
            )
            decoder_per_set = self._compute_distribution_loss(pert_cell_counts_preds, gene_targets)
            decoder_loss = decoder_per_set.mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            assert self.confidence_loss_fn is not None
            confidence_pred_vals = confidence_pred
            if confidence_pred_vals.dim() > 1:
                confidence_pred_vals = confidence_pred_vals.squeeze(-1)
            confidence_targets = per_set_main_losses.detach()
            if self.confidence_target_scale is not None:
                confidence_targets = confidence_targets * self.confidence_target_scale
            confidence_targets = confidence_targets.to(confidence_pred_vals.device)

            confidence_loss = self.confidence_weight * self.confidence_loss_fn(
                confidence_pred_vals, confidence_targets
            )
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", confidence_targets.mean())

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step logic, typically just logging the final loss and any relevant metrics."""
        if self.confidence_token is None:
            pred = self.forward(batch, padded=False)
            assert isinstance(pred, torch.Tensor)
            confidence_pred = None
        else:
            pred, confidence_pred = self.forward(batch, padded=False)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        per_set_main_losses = self._compute_distribution_loss(pred, target)
        loss = torch.nanmean(per_set_main_losses)
        self.log("test_loss", loss)

        if confidence_pred is not None:
            confidence_pred_vals = confidence_pred
            if confidence_pred_vals.dim() > 1:
                confidence_pred_vals = confidence_pred_vals.squeeze(-1)
            confidence_targets = per_set_main_losses.detach()
            if self.confidence_target_scale is not None:
                confidence_targets = confidence_targets * self.confidence_target_scale
            confidence_targets = confidence_targets.to(confidence_pred_vals.device)
            assert self.confidence_loss_fn is not None
            confidence_loss = self.confidence_weight * self.confidence_loss_fn(
                confidence_pred_vals, confidence_targets
            )
            self.log("test/confidence_loss", confidence_loss)

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, padded: bool = True, **kwargs: Any
    ):
        """Typically used for final inference. We'll replicate old logic:s returning 'preds', 'X', 'pert_name', etc."""
        if self.confidence_token is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
            confidence_pred = None
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

    def per_cell_prediction(
        self,
        test_adata: anndata.AnnData,
        perturbation_column: str,
        control_label: str,
        pert_categories: list[str],
        context_key: str | None = None,
        context_to_idx: dict[str, int] | None = None,
        basal_embedding_key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Generate predictions aligned one-to-one with `test_adata` rows."""
        # When ``basal_embedding_key`` is set, the basal (control) state fed to the
        # model is drawn from ``test_adata.obsm[basal_embedding_key]`` while the
        # returned truths/preds stay in gene space.
        labels: np.ndarray = np.asarray(test_adata.obs[perturbation_column])
        if issparse(test_adata.X):  # type: ignore[reportUnknownMemberType]
            truths = np.asarray(test_adata.X.todense(), dtype=np.float32)  # type: ignore[reportUnknownMemberType]
        else:
            truths = np.asarray(test_adata.X, dtype=np.float32)  # type: ignore[reportUnknownMemberType]
        preds = np.array(truths, dtype=np.float32, copy=True)

        ctrl_mask = labels == control_label
        # Basal source: precomputed embedding in obsm when requested, else genes.
        if basal_embedding_key is not None:
            if basal_embedding_key not in test_adata.obsm:
                raise KeyError(
                    f"basal_embedding_key='{basal_embedding_key}' not found in test_adata.obsm "
                    f"(available: {list(test_adata.obsm.keys())})."
                )
            basal_all = np.asarray(test_adata.obsm[basal_embedding_key], dtype=np.float32)
        else:
            basal_all = truths
        if basal_all.shape[1] != self.input_dim:
            raise ValueError(
                f"Basal input dim mismatch: model.input_dim={self.input_dim}, basal_dim={basal_all.shape[1]} (basal_embedding_key={basal_embedding_key!r})."
            )
        ctrl_X = basal_all[ctrl_mask, :]
        pert_to_idx = {pert: idx for idx, pert in enumerate(pert_categories)}

        # Build per-cell context indices for test data
        if (
            context_key is not None
            and context_to_idx is not None
            and context_key in test_adata.obs.columns
        ):
            cell_context_indices: np.ndarray | None = np.asarray(
                [context_to_idx.get(str(c), 0) for c in test_adata.obs[context_key]],
                dtype=np.int64,
            )
            ctrl_context_indices: np.ndarray | None = cell_context_indices[ctrl_mask]
        else:
            cell_context_indices = None
            ctrl_context_indices = None

        self.eval()
        with torch.no_grad():
            for pert in pert_categories:
                if pert == control_label:
                    continue

                pert_indices: np.ndarray = np.flatnonzero(labels == pert)
                if pert_indices.size == 0:
                    continue

                pert_onehot = np.zeros(
                    (self.cell_sentence_len, len(pert_categories)),
                    dtype=np.float32,
                )
                pert_onehot[:, pert_to_idx[pert]] = 1.0

                # Group cells by context so each sentence has a uniform context
                if cell_context_indices is not None:
                    pert_ctx = cell_context_indices[pert_indices]
                    unique_ctxs = np.unique(pert_ctx)
                    context_groups: list[tuple[np.ndarray, int]] = [
                        (pert_indices[pert_ctx == ctx], int(ctx)) for ctx in unique_ctxs
                    ]
                else:
                    context_groups = [(pert_indices, 0)]

                for group_indices, ctx_idx in context_groups:
                    for start in range(0, int(group_indices.size), self.cell_sentence_len):
                        chunk_indices = group_indices[start : start + self.cell_sentence_len]
                        n_real = int(chunk_indices.size)
                        # Sample context-matched controls when available
                        if ctrl_context_indices is not None:
                            ctx_pool = np.flatnonzero(ctrl_context_indices == ctx_idx)
                            if ctx_pool.size > 0:
                                ctrl_ids = np.random.choice(  # noqa: NPY002
                                    ctx_pool,
                                    size=self.cell_sentence_len,
                                    replace=True,
                                )
                            else:
                                ctrl_ids = np.random.choice(  # noqa: NPY002  # type: ignore[reportUnknownMemberType]
                                    ctrl_X.shape[0],
                                    size=self.cell_sentence_len,
                                    replace=True,
                                )
                        else:
                            ctrl_ids = np.random.choice(  # noqa: NPY002  # type: ignore[reportUnknownMemberType]
                                ctrl_X.shape[0],
                                size=self.cell_sentence_len,
                                replace=True,
                            )

                        batch = {
                            "ctrl_cell_emb": torch.from_numpy(ctrl_X[ctrl_ids]).to(self.device),  # type: ignore[reportUnknownMemberType]
                            "pert_emb": torch.from_numpy(pert_onehot).to(self.device),  # type: ignore[reportUnknownMemberType]
                            "batch": torch.full(
                                (self.cell_sentence_len,),
                                ctx_idx,
                                dtype=torch.long,
                                device=self.device,
                            ),
                        }
                        result = self(batch, padded=True)
                        pred_tensor = cast(
                            torch.Tensor,
                            result[0] if isinstance(result, tuple) else result,
                        )
                        preds[chunk_indices, :] = (
                            pred_tensor[:n_real]
                            .cpu()
                            .numpy()
                            .astype(
                                np.float32,
                                copy=False,
                            )
                        )

        return preds, truths, labels.astype(str).tolist()


def get_min_cells(
    adata: anndata.AnnData,
    perturbation_column: str,
    context_key: str | None = None,
) -> int:
    """
    Get the minimum number of cells across all perturbation (x context) groups.

    When context_key is provided, computes the minimum over perturbation x context
    subgroups so that cell_sentence_len is compatible with the per-context loader.
    """
    labels: np.ndarray = np.asarray(adata.obs[perturbation_column])
    if context_key is not None and context_key in adata.obs.columns:
        contexts: np.ndarray = np.asarray(adata.obs[context_key])
        min_count = int(adata.n_obs)  # start high
        for pert in np.unique(labels):
            for ctx in np.unique(contexts):
                count = int(np.sum((labels == pert) & (contexts == ctx)))
                if count > 0:
                    min_count = min(min_count, count)
        return min_count
    else:
        _, counts = np.unique(labels, return_counts=True)
        return int(counts.min())


def run_state_gene(
    train_adata: anndata.AnnData,
    valid_adata: anndata.AnnData,
    test_adata: anndata.AnnData,
    context_key: str | None,
    dataset_name: str = "dataset",
    model_dir: str = ".",
    perturbation_column: str = "perturbation",
    control_label: str = "control",
    expression_layer: str | None = None,
    device: str | None = None,
    hidden_dim: int = 64,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    transformer_backbone_key: str = "GPT2",
    transformer_backbone_kwargs: dict[str, object] | None = None,
    batch_dim: int = 10,
    use_batch_token: bool = True,
    mmd_num_chunks: int = 4,
    randomize_mmd_chunks: bool = True,
    regularization: float = 1e-5,
    epochs: int = 20,
    batch_size: int = 4,
    dataloader_num_workers: int | None = None,
    dataloader_pin_memory: bool | None = None,
    basal_embedding_key: str | None = None,
) -> dict[str, Any]:
    """
    Run the full State Transition model training and evaluation pipeline.

    Args:
        train_adata: Training AnnData (log-normalized expression in .X).
        valid_adata: Validation AnnData.
        test_adata: Test AnnData.
        context_key: .obs column name for context labels (e.g. cell type) to use as batch token classes.
        dataset_name: Name for logging/output files.
        model_dir: Directory for checkpoints and outputs.
        perturbation_column: Column in .obs with perturbation labels.
        control_label: Label for control/unperturbed cells.
        expression_layer: Optional expression layer to copy into `.X` before training.
        device: Device string (default: auto-detect).
        hidden_dim: Transformer hidden dimension.
        n_encoder_layers: Number of MLP encoder layers.
        n_decoder_layers: Number of MLP decoder layers.
        transformer_backbone_key: Transformer architecture ("GPT2" or "llama").
        transformer_backbone_kwargs: Kwargs for transformer config.
        batch_dim: Number of batch classes for batch token.
        use_batch_token: Whether to use a learnable batch token.
        mmd_num_chunks: Number of gene-dimension chunks for OT loss.
        randomize_mmd_chunks: Randomize chunk assignment each step.
        regularization: L1 regularization weight on prediction delta.
        epochs: Number of training epochs.
        batch_size: Number of perturbation sets per batch.
        dataloader_num_workers: Number of DataLoader worker processes. Defaults to a
            small nonzero value, or ``STATE_DATALOADER_NUM_WORKERS`` when set.
        dataloader_pin_memory: Whether to pin DataLoader host memory. Defaults to CUDA availability.
        basal_embedding_key: Optional key in ``.obsm`` holding a precomputed basal
            cell embedding (e.g. a Geneformer/scGPT encoder output). When set, the
            model consumes that embedding as the basal state (``input_dim`` = its
            width) and still predicts in gene space (``output_dim = gene_dim = n_vars``).

    Returns:
        Dict with full test-aligned keys: preds, truths, ctrl, pert_names.
    """
    print(f"Using context_key='{context_key}' for batch token classes.")
    if transformer_backbone_kwargs is None:
        transformer_backbone_kwargs = {"n_layer": 2, "n_head": 4, "n_embd": hidden_dim}

    train_adata = materialize_adata(train_adata, expression_layer=expression_layer)
    valid_adata = materialize_adata(valid_adata, expression_layer=expression_layer)
    test_adata = materialize_adata(test_adata, expression_layer=expression_layer)

    # Basal input dimension: the width of the precomputed embedding when provided,
    # otherwise the number of genes. Output stays in gene space regardless.
    if basal_embedding_key is not None:
        if basal_embedding_key not in train_adata.obsm:
            raise KeyError(
                f"basal_embedding_key='{basal_embedding_key}' not found in train_adata.obsm "
                f"(available: {list(train_adata.obsm.keys())})."
            )
        input_dim = int(np.asarray(train_adata.obsm[basal_embedding_key]).shape[1])
        print(
            f"Using basal_embedding_key='{basal_embedding_key}' as basal state "
            f"(input_dim={input_dim}); predicting in gene space (output_dim={train_adata.n_vars})."
        )
    else:
        input_dim = int(train_adata.n_vars)

    train_cell_sentence_len = get_min_cells(
        adata=train_adata,
        perturbation_column=perturbation_column,
        context_key=context_key,
    )
    val_cell_sentence_len = get_min_cells(
        adata=valid_adata,
        perturbation_column=perturbation_column,
        context_key=context_key,
    )
    test_cell_sentence_len = get_min_cells(
        adata=test_adata,
        perturbation_column=perturbation_column,
        context_key=context_key,
    )
    cell_sentence_len = min(train_cell_sentence_len, val_cell_sentence_len)
    print(
        f"Minimum cells per perturbation category - train: {train_cell_sentence_len},valid: {val_cell_sentence_len}, test: {test_cell_sentence_len}"
    )
    print(
        f"Using cell_sentence_len={cell_sentence_len} based on minimum cells per perturbation category across splits."
    )

    pert_categories = sorted(train_adata.obs[perturbation_column].unique().tolist())
    # Ensure all splits include all perturbation categories for consistent one-hot encoding
    for ad in [valid_adata, test_adata]:
        for p in ad.obs[perturbation_column].unique():
            if p not in pert_categories:
                pert_categories.append(p)
    pert_categories = sorted(pert_categories)

    # Compute batch_dim from context_key if provided
    if context_key is not None:
        all_contexts = sorted(
            set(
                train_adata.obs[context_key].unique().tolist()
                + valid_adata.obs[context_key].unique().tolist()
                + test_adata.obs[context_key].unique().tolist()
            )
        )
        batch_dim = len(all_contexts)
        context_to_idx: dict[str, int] = {c: i for i, c in enumerate(all_contexts)}
        logger.info(f"Using context_key='{context_key}' with {batch_dim} classes: {all_contexts}")
    else:
        context_to_idx = {}

    model = StateTransitionPerturbationModel(
        hidden_dim=hidden_dim,
        gene_dim=train_adata.n_vars,
        input_dim=input_dim,
        pert_dim=len(pert_categories),
        output_dim=train_adata.n_vars,
        embed_key=None,
        gene_names=train_adata.var_names.tolist(),
        use_batch_token=use_batch_token,
        batch_dim=batch_dim,
        transformer_backbone_key=transformer_backbone_key,
        transformer_backbone_kwargs=transformer_backbone_kwargs,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        cell_sentence_len=cell_sentence_len,
        cell_set_len=cell_sentence_len,
        extra_tokens=1 if use_batch_token else 0,
        loss_fn=SamplesLoss(loss="sinkhorn", p=2, blur=0.1),
        mmd_num_chunks=mmd_num_chunks,
        randomize_mmd_chunks=randomize_mmd_chunks,
        regularization=regularization,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=model_dir, name=dataset_name),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    train_dataloader = create_perturbation_dataloader(
        adata=train_adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        cell_sentence_len=cell_sentence_len,
        pert_categories=pert_categories,
        batch_size=batch_size,
        shuffle=True,
        context_key=context_key,
        basal_embedding_key=basal_embedding_key,
    )
    valid_dataloader = create_perturbation_dataloader(
        adata=valid_adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        cell_sentence_len=cell_sentence_len,
        pert_categories=pert_categories,
        batch_size=batch_size,
        shuffle=False,
        context_key=context_key,
        basal_embedding_key=basal_embedding_key,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    preds_np, truths_np, pert_names = model.per_cell_prediction(
        test_adata=test_adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        pert_categories=pert_categories,
        context_key=context_key,
        context_to_idx=context_to_idx,
        basal_embedding_key=basal_embedding_key,
    )

    return {"preds": preds_np, "truths": truths_np, "pert_names": pert_names}


if __name__ == "__main__":
    # Choose dataset: "synthetic", "norman19", or "replogle22"
    dataset_name = "synthetic"
    adata = anndata.AnnData()
    control_label = "control"
    perturbation_column = "perturbation"
    model_dir = "/workspaces/immunorep/immunorep-scrnaseq/data/state_gene/"
    use_counts = False
    epochs = 100
    is_synthetic = dataset_name == "synthetic"
    if is_synthetic:
        adata = generate_synthetic_perturbation_data(
            perturbation_column=perturbation_column,
            control_label=control_label,
            context_key="cell_type",
        )
        adata.layers["counts"] = adata.X.copy()  # type: ignore
    elif dataset_name == "norman19":
        adata = sc.read_h5ad(
            "/workspaces/immunorep/immunorep-scrnaseq/data/norman19/norman19_processed.h5ad"
        )
        print("Original Norman adata shape:", adata.shape)
        # indices = np.random.choice(adata.n_obs, size=1000, replace=False)
        # adata = adata[indices].copy()
    elif dataset_name == "replogle22":
        adata = sc.read_h5ad(
            "/workspaces/immunorep/immunorep-scrnaseq/data/replogle22/RPE1/processed.h5ad"
        )
        print("Original Replogle adata shape:", adata.shape)
        # indices = np.random.choice(adata.n_obs, size=1000, replace=False)
        # adata = adata[indices].copy()
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if issparse(adata.X):  # type: ignore
        data = adata.X.data  # type: ignore
        is_all_integer = np.allclose(data, np.round(data))  # type: ignore
    else:
        is_all_integer = np.allclose(adata.X, np.round(adata.X))  # type: ignore

    if use_counts and not is_all_integer:
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"]
            print("Using 'counts' layer as raw counts.")
        else:
            raise ValueError(
                "use_counts=True but adata.X contains non-integer values and no 'counts' layer exists."
            )
    elif not use_counts and is_all_integer:
        print("Data appears to be raw counts, applying normalization and log transformation...")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    elif use_counts and is_all_integer:
        print("Data contains raw counts, proceeding without modification...")
    else:
        print("Data appears to be normalized, proceeding without modification...")

    # Split adata into train (60%), valid (20%), test (20%)
    if is_synthetic:
        n = adata.n_obs
        indices = np.random.permutation(n)  # noqa
        train_end = int(0.6 * n)
        valid_end = int(0.8 * n)
        train_adata = adata[indices[:train_end]].copy()
        valid_adata = adata[indices[train_end:valid_end]].copy()
        test_adata = adata[indices[valid_end:]].copy()
    else:
        perturbations = adata.obs[perturbation_column].unique()
        print(f"Original: Unique perturbations: {len(perturbations)}")
        train_adata_list = []
        valid_adata_list = []
        test_adata_list = []
        for pert in perturbations:
            pert_adata = adata[adata.obs[perturbation_column] == pert].copy()
            n = pert_adata.n_obs
            if n < 3:
                train_adata_list.append(pert_adata)  # type: ignore
                continue
            indices = np.random.permutation(n)  # noqa
            train_end = int(0.6 * n)
            valid_end = int(0.8 * n)
            train_adata_list.append(pert_adata[indices[:train_end]])  # type: ignore
            valid_adata_list.append(pert_adata[indices[train_end:valid_end]])  # type: ignore
            test_adata_list.append(pert_adata[indices[valid_end:]])  # type: ignore
        train_adata = anndata.concat(train_adata_list)  # type: ignore
        valid_adata = anndata.concat(valid_adata_list)  # type: ignore
        test_adata = anndata.concat(test_adata_list)  # type: ignore
    print(f"Split: train={train_adata.n_obs}, valid={valid_adata.n_obs}, test={test_adata.n_obs}")
    if "cell_type" in adata.obs.columns:
        context_key = "cell_type"
    elif "cell_line" in adata.obs.columns:
        context_key = "cell_line"
    else:
        context_key = None

    # context_key = None  # for testing if context tokens are actually helpful

    out = run_state_gene(
        train_adata=train_adata,
        valid_adata=valid_adata,
        test_adata=test_adata,
        dataset_name=dataset_name,
        model_dir=model_dir,
        perturbation_column=perturbation_column,
        control_label=control_label,
        epochs=epochs,
        context_key=context_key,
    )

    is_perturbed = np.asarray(test_adata.obs[perturbation_column]) != control_label
    print(
        f"Evaluating on {is_perturbed.sum()} perturbed cells out of {len(test_adata)} total cells."
    )
    print(" unique perturbations:", len(np.unique(out["pert_names"])), len(out["pert_names"]))
    recon_mean = out["preds"][is_perturbed].mean(axis=0)
    orig_mean = out["truths"][is_perturbed].mean(axis=0)

    r, _ = pearsonr(
        orig_mean,
        recon_mean,
    )
    print(f"Pearson r={r:.3f} on {out['preds'].shape[0]} predicted perturbed cells")

    plt.figure()  # type: ignore[reportUnknownMemberType]
    plt.scatter(orig_mean, recon_mean, alpha=0.3, s=5)  # type: ignore
    plt.xlabel("Original mean expression")  # type: ignore
    plt.ylabel("Reconstructed mean expression")  # type: ignore
    plt.title(f"Per-gene mean expression (Pearson r={r:.3f})")  # type: ignore
    plt.plot([0, orig_mean.max()], [0, orig_mean.max()], "r--")  # type: ignore
    plt.savefig(model_dir + "state_gene_correlation.png")  # type: ignore
    print(f"Saved plot to {model_dir}state_gene_correlation.png")
    print("Done!")
