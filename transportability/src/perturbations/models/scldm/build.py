"""scLDM VAE and LatentDiffusion modules for Replogle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import torch

from .layers import InputTransformerVAE
from .models import VAE, LatentDiffusion
from .nnets import Decoder, DiT, Encoder
from .optimizers import AdamWLegacy
from .runtime import wsd_schedule
from .stochastic_layers import NegativeBinomialTransformerLayer
from .transport import create_transport
from .vae import TransformerVAE


def compute_training_steps(n_cells: int, batch_size: int, num_epochs: int, world_size: int = 1):
    """
    max_steps + warmup, matching upstream ``_utils.setup_datamodule_and_steps`` exactly.

    ``num_steps_per_epoch = n_cells // (batch_size * world_size)``;
    ``max_steps = num_epochs * num_steps_per_epoch``; ``warmup = int(0.1 * max_steps)``.
    """
    num_steps_per_epoch = n_cells // (batch_size * world_size)
    max_steps = max(1, num_epochs * num_steps_per_epoch)
    warmup_steps = max(1, int(0.1 * max_steps))
    return max_steps, warmup_steps


@dataclass
class VAEHParams:
    """Hyperparameters for the TransformerVAE (encoder + decoder + NB head + input layer)."""

    # scldm/experiments/configs/model/vae_base.yaml (encoder/decoder blocks)
    n_layer: int = 8
    n_inducing_points: int = 16
    n_embed: int = 32
    n_embed_latent: int = 16
    n_head: int = 8
    n_head_cross: int = 4
    dropout: float = 0.0
    bias: bool = False
    multiple_of: int = 4
    layernorm_eps: float = 1e-8
    norm_layer: str = "layernorm"
    positional_encoding: bool = True
    agg_func: Literal["log1p", "anscombe", "sqrt", "proj", "projconcat", "softbin", "log1pzero"] = (
        "log1p"  # input_layer.agg_func
    )
    shared_theta: bool = True  # decoder_name = negative_binomial_shared_theta
    # vae_optimizer (scldm.optimizers.AdamWLegacy) + vae_scheduler (wsd_schedule, sqrt decay)
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    caution: bool = False
    sched_final_lr_factor: float = 0.1
    sched_init_div_factor: int = 100
    sched_fract_decay: float = 0.1
    sched_decay_type: str = "sqrt"


@dataclass
class LDMHParams:
    """Hyperparameters for the DiT-based LDM."""

    # scldm/experiments/configs/model/ldm_base.yaml (diffusion_model = DiT)
    n_embed: int = 256
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.0
    bias: bool = True
    multiple_of: int = 4
    norm_layer: str = "layernorm"
    cfg_dropout_prob: float = 0.8
    # Replogle uses JOINT (cell_line, gene) conditioning. The base yaml default is
    # "mutually_exclusive" and the upstream replogle configs never override it, but the
    # dataset conditioning is joint AND joint size-factor sampling at generation requires
    # DiT.condition_strategy == "joint" (models._sample_log_size_factors). We default to joint
    # for Replogle; override via --condition-strategy if reproducing a mutually-exclusive dataset.
    condition_strategy: str = "joint"
    # transport (scldm.transport.create_transport)
    transport_path_type: str = "Linear"
    transport_prediction: str = "velocity"
    transport_loss_weight: str = "velocity"
    transport_train_eps: float = 1e-5
    transport_sample_eps: float = 1e-5
    # diffusion_optimizer (torch.optim.AdamW) + diffusion_scheduler (wsd_schedule, cosine)
    lr: float = 5e-4
    weight_decay: float = 0.0
    sched_final_lr_factor: float = 0.1
    sched_init_div_factor: int = 100
    sched_fract_decay: float = 1.0
    sched_decay_type: str = "cosine"
    # ema
    ema_decay: float = 0.9999
    ema_update_every: int = 10
    update_after_step: int = 10_000


def build_transformer_vae(n_genes: int, hp: VAEHParams | None = None) -> TransformerVAE:
    """Assemble the TransformerVAE (encoder + decoder + NB head + input layer)."""
    hp = hp or VAEHParams()
    encoder = Encoder(
        n_layer=hp.n_layer,
        n_inducing_points=hp.n_inducing_points,
        n_embed=hp.n_embed,
        n_embed_latent=hp.n_embed_latent,
        n_head=hp.n_head,
        n_head_cross=hp.n_head_cross,
        dropout=hp.dropout,
        bias=hp.bias,
        multiple_of=hp.multiple_of,
        layernorm_eps=hp.layernorm_eps,
        norm_layer=hp.norm_layer,
        positional_encoding=hp.positional_encoding,
    )
    decoder = Decoder(
        n_genes=n_genes,
        n_embed=hp.n_embed,
        n_embed_latent=hp.n_embed_latent,
        n_head=hp.n_head,
        n_head_cross=hp.n_head_cross,
        n_layer=hp.n_layer,
        n_inducing_points=hp.n_inducing_points,
        dropout=hp.dropout,
        bias=hp.bias,
        multiple_of=hp.multiple_of,
        layernorm_eps=hp.layernorm_eps,
        norm_layer=hp.norm_layer,
        shared_embedding=True,
        use_adaln=False,
    )

    input_layer = InputTransformerVAE(n_genes=n_genes, n_embed=hp.n_embed, agg_func=hp.agg_func)
    decoder_head = NegativeBinomialTransformerLayer(
        n_genes=n_genes,
        shared_theta=hp.shared_theta,
        n_embed=hp.n_embed,
        norm_layer=hp.norm_layer,
        layernorm_eps=hp.layernorm_eps,
    )
    return TransformerVAE(
        encoder=encoder,
        decoder=decoder,
        decoder_head=decoder_head,
        input_layer=input_layer,
    )


def build_vae_module(
    max_steps: int, warmup_steps: int, n_genes: int, hp: VAEHParams | None = None
) -> VAE:
    """The VAE LightningModule with AdamWLegacy + sqrt wsd schedule."""
    hp = hp or VAEHParams()
    vae_model = build_transformer_vae(n_genes=n_genes, hp=hp)
    vae_optimizer = partial(
        AdamWLegacy,
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        betas=hp.betas,
        caution=hp.caution,
    )
    vae_scheduler = wsd_schedule(
        num_training_steps=max_steps,
        final_lr_factor=hp.sched_final_lr_factor,
        num_warmup_steps=warmup_steps,
        init_div_factor=hp.sched_init_div_factor,
        fract_decay=hp.sched_fract_decay,
        decay_type=hp.sched_decay_type,
    )
    return VAE(vae_model=vae_model, vae_optimizer=vae_optimizer, vae_scheduler=vae_scheduler)


def build_ldm_module(
    vae_model: TransformerVAE,
    max_steps: int,
    warmup_steps: int,
    class_vocab_sizes: dict[str, int],
    vae_hp: VAEHParams | None = None,
    hp: LDMHParams | None = None,
) -> LatentDiffusion:
    """
    The LatentDiffusion LightningModule: frozen VAE tokenizer + DiT + flow-matching transport.

    ``class_vocab_sizes`` sizes the DiT condition embeddings and must be data-driven (derive it
    from ``PerturbseqDataModule.class_vocab_sizes`` at train time and persist it so generation can
    rebuild the identical architecture). Caller is responsible for loading trained VAE weights into
    ``vae_model`` and freezing it (train_ldm.py does this) before/after construction.
    """
    hp = hp or LDMHParams()
    vae_hp = vae_hp or VAEHParams()

    diffusion_model = DiT(
        n_embed=hp.n_embed,
        n_embed_input=vae_hp.n_embed_latent,  # ldm_base: ${vae.encoder.n_embed_latent}
        seq_len=vae_hp.n_inducing_points,  # ldm_base: ${vae.encoder.n_inducing_points}
        n_layer=hp.n_layer,
        n_head=hp.n_head,
        dropout=hp.dropout,
        bias=hp.bias,
        norm_layer=hp.norm_layer,
        multiple_of=hp.multiple_of,
        layernorm_eps=vae_hp.layernorm_eps,
        class_vocab_sizes=class_vocab_sizes,
        cfg_dropout_prob=hp.cfg_dropout_prob,
        condition_strategy=cast(Literal["mutually_exclusive", "joint"], hp.condition_strategy),
    )
    transport = create_transport(
        path_type=hp.transport_path_type,
        prediction=hp.transport_prediction,
        loss_weight=hp.transport_loss_weight,
        train_eps=hp.transport_train_eps,
        sample_eps=hp.transport_sample_eps,
    )
    diffusion_optimizer = partial(torch.optim.AdamW, lr=hp.lr, weight_decay=hp.weight_decay)
    diffusion_scheduler = wsd_schedule(
        num_training_steps=max_steps,
        final_lr_factor=hp.sched_final_lr_factor,
        num_warmup_steps=warmup_steps,
        init_div_factor=hp.sched_init_div_factor,
        fract_decay=hp.sched_fract_decay,
        decay_type=hp.sched_decay_type,
    )
    vae_optimizer = partial(
        AdamWLegacy,
        lr=vae_hp.lr,
        weight_decay=vae_hp.weight_decay,
        betas=vae_hp.betas,
        caution=vae_hp.caution,
    )
    return LatentDiffusion(
        vae_model=vae_model,
        vae_optimizer=vae_optimizer,
        diffusion_model=diffusion_model,
        transport=transport,
        diffusion_scheduler=diffusion_scheduler,
        diffusion_optimizer=diffusion_optimizer,
        ema_decay=hp.ema_decay,
        ema_update_every=hp.ema_update_every,
        update_after_step=hp.update_after_step,
    )
