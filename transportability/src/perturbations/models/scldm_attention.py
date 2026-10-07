"""
Exact eager-flex arithmetic for the synthetic scLDM runner on torch 2.14.0.

The vendored VAE has head dimensions below the compiled-flex minimum of 16.
Its unmasked float32 attention instead uses the math backend, whose repeated
HOP/score/mask tracing dominates these short sequences. This specialization
retains that backend's forward/backward arithmetic, including its TF32 behavior,
while omitting identity mask/score processing. Slurm experiment 918313 verified
bitwise-equal module outputs and input/parameter gradients for VAE self/cross and
DiT attention at batches 128 and 4, with about eightfold attention speedup.

Importing this module does not change scLDM. The synthetic runner explicitly
calls configure_synthetic_attention before constructing its models. Other
versions, dtypes, devices, or attention options retain the original operation.
"""

from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable
from torch.nn.attention.flex_attention import flex_attention as _original_flex_attention


def _tested_torch_version() -> bool:
    # A new torch release must be validated against its own eager implementation.
    return torch.__version__.split("+", 1)[0] == "2.14.0"


_permute_strides = None
if _tested_torch_version():
    try:
        from torch._higher_order_ops.flex_attention import _permute_strides
    except ImportError:
        pass


class _ExplicitUnmaskedAttention(torch.autograd.Function):
    """Tested math_attention/sdpa_dense_backward arithmetic, first derivatives."""

    @staticmethod
    def forward(ctx, query, key, value):
        scale = 1.0 / math.sqrt(query.size(-1))
        # The reference creates contiguous copies even for G=1 grouped heads.
        work_key = torch.repeat_interleave(key, 1, dim=1)
        work_value = torch.repeat_interleave(value, 1, dim=1)
        scores = (query @ work_key.transpose(-2, -1)) * scale
        lse_log2 = scores.logsumexp(dim=-1) / math.log(2)
        probabilities = torch._safe_softmax(scores, dim=-1)
        out = _permute_strides(probabilities @ work_value, query.stride())
        ctx.scale = scale
        ctx.save_for_backward(query, key, value, out, lse_log2)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        query, key, value, out, lse_log2 = ctx.saved_tensors
        work_key = torch.repeat_interleave(key, 1, dim=1)
        work_value = torch.repeat_interleave(value, 1, dim=1)
        logsumexp = lse_log2 * math.log(2)
        scores = (query @ work_key.transpose(-2, -1)) * ctx.scale
        probabilities = torch.exp(scores - logsumexp.unsqueeze(-1))
        grad_value = probabilities.transpose(-2, -1) @ grad_out
        grad_probabilities = grad_out @ work_value.transpose(-2, -1)
        sum_scores = torch.sum(out * grad_out, -1, keepdim=True)
        grad_scores = (probabilities * (grad_probabilities - sum_scores)) * ctx.scale
        grad_query = grad_scores @ work_key
        grad_key = grad_scores.transpose(-2, -1) @ query
        grad_key = grad_key.view(*grad_key.shape[:2], 1, *grad_key.shape[2:]).sum(2)
        grad_value = grad_value.view(*grad_value.shape[:2], 1, *grad_value.shape[2:]).sum(2)
        return (
            _permute_strides(grad_query, query.stride()),
            _permute_strides(grad_key, key.stride()),
            _permute_strides(grad_value, value.stride()),
        )


def synthetic_flex_attention(query, key, value, *args, **kwargs):
    """Use the validated specialization, delegating every other call unchanged."""
    default_options = (
        not args
        and kwargs.keys() <= {"block_mask", "score_mod", "return_lse"}
        and kwargs.get("block_mask") is None
        and kwargs.get("score_mod") is None
        and kwargs.get("return_lse", False) is False
    )
    tensors = (query, key, value)
    supported = (
        default_options
        and _tested_torch_version()
        and _permute_strides is not None
        and not torch.compiler.is_compiling()
        and not torch.is_autocast_enabled()
        and all(
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 4
            and tensor.dtype == torch.float32
            and tensor.is_cuda
            for tensor in tensors
        )
        and query.device == key.device == value.device
        and query.shape[:2] == key.shape[:2] == value.shape[:2]
        and key.shape == value.shape
        and query.shape[-1] == key.shape[-1]
        and all(size > 0 for tensor in tensors for size in tensor.shape)
    )
    if not supported:
        return _original_flex_attention(query, key, value, *args, **kwargs)
    return _ExplicitUnmaskedAttention.apply(query, key, value)


def configure_synthetic_attention() -> bool:
    """Idempotently configure attention for models subsequently built here."""
    if not _tested_torch_version() or _permute_strides is None:
        return False
    from .scldm import layers

    layers.flex_attention = synthetic_flex_attention
    return True
