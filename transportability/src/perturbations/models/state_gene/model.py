"""Defines the StateEmbeddingModel, a transformer-based model for predicting gene expression perturbations from cell state embeddings."""

import logging
import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, LRScheduler

from .flash_transformer import (
    FlashTransformerEncoder,
    FlashTransformerEncoderLayer,
)
from .loss import (
    KLDivergenceLoss,
    MMDLoss,
    TabularLoss,
    WassersteinLoss,
)
from .utils import get_dataset_cfg, get_embedding_cfg


class SkipBlock(nn.Module):
    """A simple skip block with two linear layers and a residual connection."""

    def __init__(self, in_features: int):
        """

        Given input X of size in_features.

        - out = layernorm(x + MLP(MLP(X))).
        """
        super().__init__()  # type: ignore
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features * 2, bias=True)
        self.dense = nn.Linear(in_features * 2, in_features, bias=True)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the skip block."""
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x


def nanstd(x: torch.Tensor) -> torch.Tensor:
    """Compute the standard deviation of a tensor while ignoring NaN values."""
    return torch.sqrt(
        torch.nanmean(torch.pow(x - torch.nanmean(x, dim=-1).unsqueeze(-1), 2), dim=-1)
    )


class StateEmbeddingModel(L.LightningModule):
    """A transformer-based model for predicting gene expression perturbations from cell state embeddings."""

    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        warmup_steps: int = 0,
        compiled: bool = False,
        max_lr: float = 4e-4,
        emb_cnt: int = 145469,
        emb_size: int = 5120,
        cfg: DictConfig | None = None,
        collater: object | None = None,
    ):
        """Initializes the StateEmbeddingModel."""
        super().__init__()
        if cfg is None:
            raise ValueError("StateEmbeddingModel requires a cfg (DictConfig).")
        self.cfg = cfg
        self.save_hyperparameters()
        self.compiled = compiled
        self.model_type = "Transformer"
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))

        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        self.collater = collater
        # Encodes Tokens
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model, bias=True),
            nn.LayerNorm(d_model),  # Moved before activation
            nn.SiLU(),  # Changed to SiLU
        )

        # Create a list of FlashTransformerEncoderLayer instances
        layers: list[nn.Module] = [
            FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout)
            for _ in range(nlayers)
        ]
        self.transformer_encoder = FlashTransformerEncoder(layers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)  # type: ignore[assignment]

        self.d_model = d_model
        self.dropout = dropout

        self.decoder = nn.Sequential(
            SkipBlock(d_model),
            nn.Linear(d_model, output_dim, bias=True),
        )

        if compiled:
            self.decoder = torch.compile(self.decoder)  # type: ignore[assignment]

        self.z_dim_rd = 1 if self.cfg.model.rda else 0
        self.z_dim_ds = 10 if self.cfg.model.get("dataset_correction", False) else 0
        self.z_dim = self.z_dim_rd + self.z_dim_ds

        self.binary_decoder = nn.Sequential(
            SkipBlock(output_dim + d_model + self.z_dim),
            SkipBlock(output_dim + d_model + self.z_dim),
            nn.Linear(output_dim + d_model + self.z_dim, 1, bias=True),
        )

        if self.cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(
                nn.Linear(1, 512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(512, 10),
            )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)  # type: ignore[assignment]

        # Encodes Tokens for Decoder
        self.gene_embedding_layer = self.encoder  # reuse this layer

        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)  # type: ignore[assignment]

        self.pe_embedding = None  # TODO: make this cleaner for the type checker, right now it gets set externally after model init
        self.step_ctr = 0

        self.true_top_genes: dict[str, torch.Tensor] | None = None
        self.protein_embeds: dict[str, torch.Tensor] | None = None

        self._last_val_de_check = 0
        self._last_val_perturbation_check = 0

        if getattr(self.cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
            self.dataset_embedder = nn.Linear(output_dim, self.z_dim_ds)

            # Assume self.cfg.model.num_datasets is set to the number of unique datasets.
            num_dataset = get_dataset_cfg(self.cfg).num_datasets
            self.dataset_encoder = nn.Sequential(
                nn.Linear(output_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_dataset),
            )

            # this should be a classification label loss
            self.dataset_loss = nn.CrossEntropyLoss()
        else:
            self.dataset_token = None

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Persist a snapshot of the training config inside the checkpoint so downstream inference/eval can run without an external config file."""
        try:
            # Store both a YAML snapshot and a resolved container for robustness
            checkpoint["cfg_yaml"] = OmegaConf.to_yaml(self.cfg)
        except Exception:
            # Never block checkpointing if config serialization fails
            pass

        # Also package protein embeddings for standalone inference/transform.
        try:
            if self.protein_embeds is not None:
                pe = self.protein_embeds
            else:
                # Load from configured path as a fallback

                pe: dict[str, torch.Tensor] = torch.load(
                    get_embedding_cfg(self.cfg).all_embeddings,
                    map_location="cpu",
                    weights_only=False,
                )
                # Ensure CPU tensors in the dictionary

            cpu_pe = {}
            for k, v in pe.items():
                try:
                    cpu_pe[k] = (
                        v.detach().to("cpu")
                        if hasattr(v, "detach")
                        else torch.tensor(v, device="cpu")
                    )
                except Exception:
                    cpu_pe[k] = v
                checkpoint["protein_embeds_dict"] = cpu_pe
        except Exception:
            # Do not block checkpoint save if embedding packaging fails
            pass

    def _compute_embedding_for_batch(self, batch: tuple[torch.Tensor | None, ...]):
        assert (
            batch[0] is not None
            and batch[1] is not None
            and batch[2] is not None
            and batch[5] is not None
        ), "Batch tensors cannot be None"
        batch_sentences = batch[0].to(self.device)
        task_genes = batch[1].to(self.device)
        targets = batch[2].to(self.device)
        batch_weights = batch[4]
        mask = batch[5]
        mask = mask.to(torch.bool)
        batch_sentences_counts = batch[7]
        if batch_sentences_counts is not None:
            batch_sentences_counts = batch_sentences_counts.to(self.device)
        dataset_nums = batch[8]
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(self.device)

        # convert the cell sentence and task sentence into embeddings
        assert self.pe_embedding is not None, (
            "Positional embedding layer must be set before calling forward"
        )
        batch_sentences = self.pe_embedding(batch_sentences)
        task_genes = self.pe_embedding(task_genes)

        # Normalize token outputs now
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)

        # Add a learnable CLS token to the beginning of the sentence
        batch_sentences[:, 0, :] = self.cls_token.expand(batch_sentences.size(0), -1)

        # Optionally add a learnable dataset token to the end of the sentence
        if self.dataset_token is not None:
            dataset_token = self.dataset_token.expand(batch_sentences.size(0), -1).unsqueeze(1)
            batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)
            # concatenate a False to the mask on dim 1
            mask = torch.cat((mask, torch.zeros(mask.size(0), 1, device=mask.device).bool()), dim=1)

        # mask out the genes embeddings that appear in the task sentence
        _, embedding, dataset_emb = self.forward(
            batch_sentences, mask=mask, counts=batch_sentences_counts, dataset_nums=dataset_nums
        )

        task_genes = self.gene_embedding_layer(task_genes)
        return task_genes, targets, batch_weights, embedding, dataset_emb

    def get_gene_embedding(self, genes: list[str]) -> torch.Tensor:
        """Given a list of gene names, return their corresponding embeddings."""
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(
                get_embedding_cfg(self.cfg).all_embeddings, weights_only=False
            )
        assert self.protein_embeds is not None
        protein_embeds_list: list[torch.Tensor] = [
            self.protein_embeds[x]
            if x in self.protein_embeds
            else torch.zeros(get_embedding_cfg(self.cfg).size)
            for x in genes
        ]
        protein_embeds_stacked = torch.stack(protein_embeds_list).to(self.device)
        if protein_embeds_stacked.sum() == 0:
            raise ValueError("No gene embeddings found")

        return self.gene_embedding_layer(protein_embeds_stacked)

    @staticmethod
    def resize_batch(
        cell_embeds: torch.Tensor,
        task_embeds: torch.Tensor,
        task_counts: torch.Tensor | None = None,
        sampled_rda: torch.Tensor | None = None,
        ds_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resize the task embedding to match the cell embedding dimensions and concatenate them."""
        task_embeds_repeated = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)
        cell_embedding_batches = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)
        if sampled_rda is not None:
            # computes mu and std dev from Y
            reshaped_counts = sampled_rda.unsqueeze(1)
            reshaped_counts = reshaped_counts.repeat(1, task_embeds_repeated.shape[1], 1)
            combine = torch.cat(
                (task_embeds_repeated, cell_embedding_batches, reshaped_counts), dim=2
            )
        elif task_counts is not None:
            reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, task_embeds_repeated.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat(
                (task_embeds_repeated, cell_embedding_batches, reshaped_counts), dim=2
            )
        else:
            # Original behavior if total_counts is None
            combine = torch.cat((task_embeds_repeated, cell_embedding_batches), dim=2)

        if ds_emb is not None:
            # ds_emb is a tensor of shape (batch_size, 10). concatenate it to the combine tensor
            ds_emb = ds_emb.unsqueeze(1).repeat(1, task_embeds_repeated.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        return combine

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor,
        counts: torch.Tensor | None = None,
        dataset_nums: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the model.

        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken].
            mask: Boolean Tensor, shape [batch_size, seq_len], where True indicates valid tokens and False indicates padding.
            counts: Tensor of shape [batch_size, seq_len] containing gene expression counts for each token, if count encoding is enabled.
            dataset_nums: Tensor of shape [batch_size] containing dataset identifiers for each sample, if dataset correction is enabled.

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        if counts is not None:
            # scFoundation-style soft binning for counts
            counts = counts.unsqueeze(-1)  # now B x H x 1

            # Step 1: Transform count values into bin distribution
            bin_weights = self.count_encoder(counts)  # B x H x 10
            bin_weights = F.softmax(bin_weights, dim=-1)  # Convert to probabilities over bins

            # Step 2: Get bin embeddings
            bin_indices = torch.arange(10, device=self.device)  # 10 bins
            bin_embeddings = self.bin_encoder(bin_indices)  # 10 x d_model

            # Step 3: Compute weighted sum of bin embeddings
            count_emb = torch.matmul(bin_weights, bin_embeddings)

            if self.dataset_token is not None:
                # append B x 1 x d_model to count_emb of all zeros
                dataset_count_emb = torch.zeros(
                    count_emb.size(0), 1, count_emb.size(2), device=self.device
                )
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)  # B x H x d_model

            # Add count embeddings to token embeddings
            src = (
                src + count_emb
            )  # should both be B x H x self.d_model, or B x H + 1 x self.d_model if dataset correction

        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)  # batch x seq_len x 128
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :]  # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1)  # Normalize.

        # we must be in train mode to use dataset correction
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        return gene_output, embedding, dataset_emb

    def _log_nonzero_elements_stats(self, batch_sentences: torch.Tensor, prefix: str = "trainer"):
        """
        Track and log non-zero elements in the sentence for ablation study.

        This function analyzes the mask tensor to count how many positions in each cell sentence
        contain expressed genes (non-zero) versus unexpressed/padded genes (zero). This is useful
        for understanding how padding with unexpressed gene embeddings affects model learning.

        Args:
            batch_sentences: Boolean tensor of shape (batch_size, seq_len) where True indicates
                  non-zero (expressed) genes and False indicates zero (unexpressed/padded) genes.
            prefix: String prefix for logging (e.g., "trainer" or "validation")

        Logs:
            - {prefix}/avg_nonzero_genes: Average number of non-zero genes per cell
            - {prefix}/nonzero_fraction: Fraction of non-zero genes relative to total slots
        """
        # Count non-zero elements per cell in the batch
        nonzero_counts = batch_sentences.sum(dim=1)  # Sum across sequence dimension
        avg_nonzero = nonzero_counts.float().mean().item()

        # Calculate the fraction of non-zero elements (excluding CLS token)
        total_slots = batch_sentences.shape[1]
        nonzero_fraction = avg_nonzero / total_slots

        # Log the statistics
        self.log(f"{prefix}/avg_nonzero_genes", avg_nonzero)
        self.log(f"{prefix}/nonzero_fraction", nonzero_fraction)

    def shared_step(self, batch: tuple[torch.Tensor | None, ...], batch_idx: int) -> torch.Tensor:
        """Shared logic for training and validation steps."""
        logging.info(f"Step {self.global_step} - Batch {batch_idx}")
        target_genes, targets, batch_weights, embs, dataset_embs = (
            self._compute_embedding_for_batch(batch)
        )
        assert batch[7] is not None and batch[8] is not None, "Batch sentences cannot be None"

        # Track non-zero elements in the sentence
        batch_sentences = batch[7].to(self.device).bool()
        prefix = "trainer" if self.training else "validation"
        self._log_nonzero_elements_stats(batch_sentences, prefix)

        z = embs.unsqueeze(1).repeat(1, target_genes.shape[1], 1)  # CLS token

        if self.z_dim_rd == 1:
            if self.cfg.model.rda:
                mu = torch.nan_to_num(
                    torch.nanmean(
                        targets.float().masked_fill(targets == 0, float("nan")),
                        dim=1,
                    ),
                    nan=0.0,
                )
                reshaped_counts = mu.unsqueeze(1).unsqueeze(2)
                reshaped_counts = reshaped_counts.repeat(1, target_genes.shape[1], 1)
                combine = torch.cat((target_genes, z, reshaped_counts), dim=2)
            else:
                combine = torch.cat((target_genes, z), dim=2)
        else:
            assert self.z_dim_rd == 0
            combine = torch.cat((target_genes, z), dim=2)

        if self.dataset_token is not None and dataset_embs is not None:
            ds_emb = self.dataset_embedder(dataset_embs)
            ds_emb = ds_emb.unsqueeze(1).repeat(1, target_genes.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        # concatenate the counts
        decs = self.binary_decoder(combine)

        if self.cfg.loss.name == "cross_entropy":
            criterion = BCEWithLogitsLoss()
            target = targets
        elif self.cfg.loss.name == "mse":
            criterion = nn.MSELoss()
            target = targets
        elif self.cfg.loss.name == "wasserstein":
            criterion = WassersteinLoss()
            target = targets
        elif self.cfg.loss.name == "kl_divergence":
            criterion = KLDivergenceLoss(apply_normalization=self.cfg.loss.normalization)
            target = batch_weights
        elif self.cfg.loss.name == "mmd":
            kernel = self.cfg.loss.get("kernel", "energy")
            criterion = MMDLoss(
                kernel=kernel, downsample=self.cfg.model.num_downsample if self.training else 1
            )
            target = targets
        elif self.cfg.loss.name == "tabular":
            criterion = TabularLoss(
                shared=self.cfg.dataset.S,
                downsample=self.cfg.model.num_downsample if self.training else 1,
            )
            target = targets
        else:
            raise ValueError(f"Loss {self.cfg.loss.name} not supported")

        loss: torch.Tensor = criterion(decs.squeeze(), target)
        if dataset_embs is not None:
            # use the dataset loss
            dataset_pred = self.dataset_encoder(dataset_embs)  # B x # datasets
            dataset_labels = batch[8].to(self.device).long()

            # self.dataset_loss is a nn.CrossEntropyLoss
            dataset_loss = self.dataset_loss(dataset_pred, dataset_labels)
            if self.training:
                self.log("trainer/dataset_loss", dataset_loss)
                loss = loss + dataset_loss
            else:
                self.log("validation/dataset_loss", dataset_loss)

        sch = self.lr_schedulers()
        assert isinstance(sch, ChainedScheduler)
        sch.step()

        # for scheduler in sch._schedulers:
        #     if isinstance(scheduler, ChainedScheduler):
        #         scheduler.step(loss)
        #     else:
        #         scheduler.step()
        # sch._last_lr = [group["lr"] for group in sch._schedulers[-1].optimizer.param_groups]
        return loss

    @torch.compile(disable=True)  # type: ignore[method-assign]
    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int):
        """Training step logic."""
        loss = self.shared_step(batch, batch_idx)
        self.log("trainer/train_loss", loss)
        return loss

    @torch.compile(disable=True)  # type: ignore[method-assign]
    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int):
        """Validation step logic."""
        loss = self.shared_step(batch, batch_idx)
        self.log("validation/val_loss", loss)
        return loss

    def configure_optimizers(self) -> dict[str, object]:  # type: ignore[override]
        """Configure optimizers and learning rate schedulers."""
        max_lr = self.max_lr
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=max_lr, weight_decay=self.cfg.optimizer.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches * 2  # not sure why need to do this

        lr_schedulers = [
            LinearLR(
                optimizer,
                start_factor=self.cfg.optimizer.start,
                end_factor=self.cfg.optimizer.end,
                total_iters=int(0.03 * total_steps),
            )
        ]
        # lr_schedulers.append(CosineAnnealingLR(optimizer, eta_min=max_lr * 0.3, T_max=total_steps))

        lr_schedulers: list[LRScheduler] = [
            LinearLR(
                optimizer,
                start_factor=self.cfg.optimizer.start,
                end_factor=self.cfg.optimizer.end,
                total_iters=int(0.03 * total_steps),
            )
        ]
        scheduler = ChainedScheduler(lr_schedulers)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }

    def update_config(self, new_cfg: DictConfig):
        """Update the model's config after loading from checkpoint."""
        self.cfg = new_cfg
