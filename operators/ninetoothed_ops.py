"""Thin adapter between the Llama operator API and NineToothed kernels."""

import functools
import importlib
import warnings

import torch

from .registry import register_operator


@functools.lru_cache(maxsize=1)
def _load_rms_norm_kernel():
    module = importlib.import_module(
        ".ninetoothed_kernels.fused_rms_norm", package=__package__
    )
    return module.kernel


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if input.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight size must match the final input dimension")
    if not input.is_contiguous() or not weight.is_contiguous():
        raise ValueError("NineToothed rms_norm expects contiguous tensors")

    hidden_size = input.shape[-1]
    input_2d = input.reshape(-1, hidden_size)
    weight_2d = weight.expand_as(input_2d)
    output_2d = torch.empty_like(input_2d)
    _load_rms_norm_kernel()(
        input_2d,
        weight_2d,
        float(eps),
        output_2d,
        BLOCK_SIZE=hidden_size,
    )
    return output_2d.reshape(input.shape)


def rope(input, sin_table, cos_table):
    """Temporary reference path until a student supplies the NineToothed kernel."""
    warnings.warn(
        "NineToothed rope is not implemented; using the PyTorch reference. "
        "Replace operators.ninetoothed_ops.rope for performance work.",
        RuntimeWarning,
        stacklevel=2,
    )
    from .torch_ops import rope as torch_rope

    return torch_rope(input, sin_table, cos_table)


register_operator("ninetoothed", "rms_norm", rms_norm)
register_operator("ninetoothed", "rope", rope, fallback_to_torch=True)
