"""
Neural network layers for the scLDM model.

Adapted from:
- https://github.com/epfml/llm-baselines/blob/main/src/models
- https://github.com/pytorch-labs/attention-gym/blob/main/examples/
"""

import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

SCORE_MOD = {
    "noop": None,
}


NORM_LAYERS = {
    "layernorm": nn.LayerNorm,
}


def log1p_transform(
    genes: torch.Tensor, counts: torch.Tensor, zero_encoding: bool = False
) -> torch.Tensor:
    """Applies a log1p transformation to the counts and optionally encodes zeros as -1."""
    if zero_encoding:
        return genes * torch.where(
            counts == 0, torch.tensor(-1.0, device=counts.device), torch.log1p(counts)
        )
    return genes * torch.log1p(counts)


def asinh_sqrt_transform(genes: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Applies an asinh(sqrt(counts + 1)) transformation to the counts."""
    counts = torch.asinh(torch.sqrt(counts + 1.0))
    return genes * counts


def sqrt_transform(genes: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Applies a sqrt(counts + 1) transformation to the counts."""
    counts = torch.sqrt(counts + 1.0)
    return genes * counts


class Projection(nn.Module):
    """Projects counts into the embedding space using a linear layer and adds it to the gene embeddings."""

    def __init__(self, n_embed: int):
        """Initialize the projection layer with the specified embedding dimension."""
        super().__init__()
        self.count_embedding = nn.Linear(1, n_embed)

    def forward(self, genes: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Forward pass: project counts and add to gene embeddings."""
        counts = self.count_embedding(counts)
        return genes + counts


class ProjectionConcat(nn.Module):
    """Concatenates gene embeddings with log1p-transformed counts and projects back to embedding space."""

    def __init__(self, n_embed: int):
        """Initialize the projection layer with the specified embedding dimension."""
        super().__init__()
        self.mix = nn.Linear(n_embed * 2, n_embed)

    def forward(self, genes: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Forward pass: concatenate gene embeddings with log1p-transformed counts and project."""
        # More efficient: expand counts to match genes shape
        log_counts = torch.log1p(counts).expand(-1, -1, genes.shape[-1])
        return self.mix(torch.cat([genes, log_counts], dim=-1))


class SoftBinProjection(nn.Module):
    """Projects counts into the embedding space using soft binning and adds it to the gene embeddings."""

    def __init__(self, n_embed: int, n_bins: int = 10, hidden_dim: int = 64):
        """Initialize the soft bin projection layer with the specified embedding dimension and number of bins."""
        super().__init__()
        self.n_bins = n_bins
        self.mlp_count = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.n_bins)
        )
        self.bin_embeddings = nn.Parameter(torch.randn(self.n_bins, n_embed))

    def forward(self, genes: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Forward pass: project counts using soft binning and add to gene embeddings."""
        bin_logits = self.mlp_count(counts)  # (..., n_bins)
        bin_weights = torch.softmax(bin_logits, dim=-1)  # (..., n_bins)
        count_embedding = torch.einsum("...k,kd->...d", bin_weights, self.bin_embeddings)
        return genes + count_embedding


PROJ_FUNC = {
    "log1p": log1p_transform,
    "log1pzero": partial(log1p_transform, zero_encoding=True),
    "anscombe": asinh_sqrt_transform,
    "sqrt": sqrt_transform,
    "proj": Projection,
    "projconcat": ProjectionConcat,
    "softbin": SoftBinProjection,
}


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies adaptive layer normalization modulation."""
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


class InputTransformerVAE(nn.Module):
    """Transforms input counts and gene indices into embeddings suitable for the VAE, using a specified aggregation function."""

    def __init__(
        self,
        n_genes: int,
        n_embed: int,
        agg_func: Literal[
            "log1p", "anscombe", "sqrt", "proj", "projconcat", "softbin", "log1pzero"
        ],
    ):
        """Initialize the input transformer for the VAE with the specified number of genes, embedding dimension, and aggregation function."""
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes + 1, n_embed)

        self.projection: nn.Module | Callable[..., Any] = PROJ_FUNC[agg_func]
        if agg_func in ["proj", "projconcat", "softbin"]:
            self.projection = cast("type[nn.Module]", PROJ_FUNC[agg_func])(n_embed)

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: embed genes and project counts using the specified aggregation function."""
        genes_emb = self.gene_embedding(genes)
        output = self.projection(genes_emb, counts.unsqueeze(-1))
        return output


class SelfAttention(nn.Module):
    """Multi-head self-attention module with optional flexible attention mechanism."""

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ):
        """Initialize the multi-head self-attention module with the specified number of heads, embedding dimension, dropout, and bias."""
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(self.dropout)
        self.flex_attention = flex_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the multi-head self-attention module."""
        B, S, D = x.shape

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, S, self.n_head, D // self.n_head)
        q = q.view(B, S, self.n_head, D // self.n_head)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.view(B, S, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nH, S, hD)

        y: torch.Tensor = self.flex_attention(
            q, k, v, block_mask=None, score_mod=None, return_lse=False
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, S, D)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward MLP module used within the transformer block."""

    def __init__(self, n_embed: int, multiple_of: int):
        """Initialize the MLP with the specified embedding dimension and multiple_of parameter for hidden layer sizing."""
        super().__init__()

        hidden_dim = n_embed * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, n_embed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP module. Applies a linear transformation, SiLU activation, element-wise multiplication with another linear transformation, and a final projection."""
        return self.c_proj(nn.functional.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Transformer block consisting of a self-attention module followed by an MLP, with optional AdaLN modulation."""

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        use_adaln: bool = False,
        elementwise_affine: bool = True,
    ):
        """Initialize the transformer block with the specified parameters, including embedding dimension, number of heads, dropout, bias, normalization layer, multiple_of for MLP, layernorm epsilon, and optional AdaLN modulation."""
        super().__init__()

        self.ln_1 = NORM_LAYERS[norm_layer](
            n_embed, eps=layernorm_eps, elementwise_affine=elementwise_affine
        )
        self.ln_2 = NORM_LAYERS[norm_layer](
            n_embed, eps=layernorm_eps, elementwise_affine=elementwise_affine
        )

        self.attn = SelfAttention(
            n_embed=n_embed,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
        )

        self.mlp = MLP(n_embed=n_embed, multiple_of=multiple_of)

        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(n_embed, 6 * n_embed, bias=True)
            )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the transformer block. Applies self-attention and MLP with optional AdaLN modulation based on the condition tensor."""
        if self.use_adaln:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_modulation(condition).chunk(6, dim=-1)
            )
            norm1_out = modulate(self.ln_1(x), scale_attn, shift_attn)
            x = x + gate_attn * self.attn(norm1_out)
            norm2_out = modulate(self.ln_2(x), scale_mlp, shift_mlp)
            x_ = gate_mlp * self.mlp(norm2_out)
            x = x + x_
        else:
            x = x + self.attn(self.ln_1(x))
            x_ = self.mlp(self.ln_2(x))
            x = x + x_
        return x


class CrossAttention(nn.Module):
    """Cross-attention module that computes attention between a set of input embeddings and a set of query embeddings."""

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ):
        """Initialize the cross-attention module with the specified embedding dimension, number of heads, dropout, and bias."""
        super().__init__()

        self.n_head = n_head
        self.n_embed = n_embed

        self.c_attn = nn.Linear(n_embed, 2 * n_embed, bias=bias)  # key, value projections for x
        self.c_attn_q = nn.Linear(n_embed, n_embed, bias=bias)  # key, projection for q
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)  # output projection
        self.flex_attention = flex_attention
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Forward pass for the cross-attention module. Computes attention between input embeddings x and query embeddings q, returning the pooled output after attention and projection."""
        B, S, _ = x.shape
        _, M, Dout = q.shape  # get seq_len of inducing points

        k, v = self.c_attn(x).split(self.n_embed, dim=-1)
        q = self.c_attn_q(q)

        k = k.view(B, S, self.n_head, Dout // self.n_head)
        v = v.view(B, S, self.n_head, Dout // self.n_head)
        q = q.view(B, M, self.n_head, Dout // self.n_head)
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, nH, S, hD)

        y: torch.Tensor = self.flex_attention(
            q, k, v, block_mask=None, score_mod=None, return_lse=False
        )
        y = y.transpose(1, 2).contiguous().view(B, M, Dout)
        pooled_x = self.resid_dropout(self.c_proj(y))  # B, M, Dout

        return pooled_x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block that applies cross-attention between input embeddings and query embeddings, followed by an MLP, with optional AdaLN modulation."""

    def __init__(
        self,
        n_embed: int,
        n_inducing_points: int,
        n_head: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        use_adaln: bool = False,
    ):
        """Initialize the cross-attention block with the specified parameters, including embedding dimension, number of inducing points, number of heads, dropout, bias, normalization layer, multiple_of for MLP, layernorm epsilon, and optional AdaLN modulation."""
        super().__init__()

        self.inducing_points = (
            None
            if n_inducing_points == 0
            else nn.Parameter(torch.randn(n_inducing_points, n_embed), requires_grad=True)
        )

        self.ln_1 = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        self.ln_1q = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)

        self.attn = CrossAttention(
            n_embed=n_embed,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
        )

        self.ln_2 = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        self.mlp = MLP(n_embed=n_embed, multiple_of=multiple_of)
        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(n_embed, 6 * n_embed, bias=True)
            )
            self.adaln_modulation_q = nn.Sequential(
                nn.SiLU(), nn.Linear(n_embed, 2 * n_embed, bias=True)
            )

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the cross-attention block. Applies cross-attention between input embeddings x and query embeddings q, followed by an MLP, with optional AdaLN modulation based on the condition tensor."""
        B, _, _ = x.shape
        if self.inducing_points is not None and q is None:
            q = self.inducing_points.expand(
                B, -1, -1
            )  # expand inducing points over batches without allocation
        if self.use_adaln:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_modulation(condition).chunk(6, dim=-1)
            )
            shift_q, scale_q = self.adaln_modulation_q(condition).chunk(2, dim=-1)
            norm1_xout = modulate(self.ln_1(x), scale_attn, shift_attn)
            norm1_qout = modulate(self.ln_1q(q), scale_q, shift_q)
            x = q + gate_attn * self.attn(norm1_xout, norm1_qout)
            norm2_out = modulate(self.ln_2(x), scale_mlp, shift_mlp)
            x_ = gate_mlp * self.mlp(norm2_out)
            x = x + x_
        else:
            attn_output = self.attn(self.ln_1(x), self.ln_1q(q))
            x = q + attn_output
            x_ = self.mlp(self.ln_2(x))
            x = x + x_
        return x

    def extra_repr(self):
        """Provides a string representation of the cross-attention block, showing the shape of the inducing points parameter."""
        shape = None if self.inducing_points is None else tuple(self.inducing_points.shape)
        return f"(inducing_points): Parameter(shape={shape})"


##########################
#     Layers for DiT     #
##########################
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """Initialize the timestep embedder with the specified hidden size and frequency embedding size."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Creates sinusoidal timestep embeddings for the given timesteps t and embedding dimension dim, with an optional maximum period for the frequencies."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the timestep embedding for the given timesteps t and passes it through the MLP to obtain the final embedding."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def get_1d_sincos_pos_embed(embed_dim: int, seq_len: int):
    """Generates 1D sinusoidal positional embeddings for a sequence of length seq_len and embedding dimension embed_dim."""
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    positions = np.arange(seq_len, dtype=np.float32)

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (embed_dim // 2,)

    pos = positions.reshape(-1, 1)  # (seq_len, 1)
    omega = omega.reshape(1, -1)  # (1, embed_dim // 2)

    out = pos * omega  # (seq_len, embed_dim // 2)

    emb_sin = np.sin(out)  # (seq_len, embed_dim // 2)
    emb_cos = np.cos(out)  # (seq_len, embed_dim // 2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (seq_len, embed_dim)

    return emb


class FinalLayerDit(nn.Module):
    """The final layer of DiT."""

    def __init__(self, n_embed: int, n_embed_input: int, bias: bool, layernorm_eps: float):
        """Initialize the final layer of DiT with the specified embedding dimensions, bias, and layer normalization epsilon."""
        super().__init__()
        self.norm_final = nn.LayerNorm(n_embed, elementwise_affine=False, eps=layernorm_eps)
        self.linear = nn.Linear(n_embed, n_embed_input, bias=bias)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(n_embed, 2 * n_embed, bias=bias))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass for the final layer of DiT. Applies AdaLN modulation using the condition tensor c, followed by layer normalization and a linear projection."""
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
