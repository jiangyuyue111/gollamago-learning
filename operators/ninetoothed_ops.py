"""Thin adapter between the Llama operator API and NineToothed kernels."""

import functools
import importlib
import warnings

import torch

from .registry import OperatorUnavailableError, register_operator


@functools.lru_cache(maxsize=1)
def _load_rms_norm_kernel():
    try:
        module = importlib.import_module(
            ".ninetoothed_kernels.fused_rms_norm", package=__package__
        )
    except (ImportError, OSError) as error:
        raise OperatorUnavailableError(
            "ninetoothed RMSNorm requires the optional 'ninetoothed' package "
            "and a compatible compiler backend"
        ) from error
    return module.kernel


def _apply_rms_norm_kernel(kernel, input, weight, eps):
    hidden_size = input.shape[-1]
    input_2d = input.view(-1, hidden_size)
    weight_2d = weight.expand_as(input_2d)
    output_2d = torch.empty_like(input_2d)
    kernel(
        input_2d,
        weight_2d,
        float(eps),
        output_2d,
        BLOCK_SIZE=hidden_size,
    )
    return output_2d.view_as(input)


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if input.ndim == 0:
        raise ValueError("ninetoothed rms_norm expects at least one input dimension")
    if weight.shape != (input.shape[-1],):
        raise ValueError("RMSNorm weight must match the final input dimension")
    if input.device != weight.device or input.dtype != weight.dtype:
        raise ValueError("RMSNorm input and weight must share device and dtype")
    if not input.is_contiguous() or not weight.is_contiguous():
        raise ValueError("ninetoothed rms_norm expects contiguous tensors")
    if input.device.type != "cuda":
        raise RuntimeError(
            "ninetoothed rms_norm requires a CUDA-compatible accelerator tensor"
        )
    return _apply_rms_norm_kernel(_load_rms_norm_kernel(), input, weight, eps)


def rope(input, sin_table, cos_table):
    warnings.warn(
        "NineToothed rope is not implemented; using the PyTorch reference. "
        "This fallback must not be included in backend performance claims.",
        RuntimeWarning,
        stacklevel=2,
    )
    from .torch_ops import rope as torch_rope

    return torch_rope(input, sin_table, cos_table)


register_operator("ninetoothed", "rms_norm", rms_norm)
register_operator("ninetoothed", "rope", rope, fallback_to_torch=True)
