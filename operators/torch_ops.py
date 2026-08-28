"""Reference PyTorch operators used for correctness and baseline measurements."""

import torch

from .registry import register_operator


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return input * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rope(input: torch.Tensor, sin_table: torch.Tensor, cos_table: torch.Tensor) -> torch.Tensor:
    """Apply Llama half-rotation to [..., sequence, heads, head_dim] tensors."""
    half = input.shape[-1] // 2
    sin = sin_table[: input.shape[-3], :half].unsqueeze(0).unsqueeze(2)
    cos = cos_table[: input.shape[-3], :half].unsqueeze(0).unsqueeze(2)
    first, second = input[..., :half], input[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


register_operator("torch", "rms_norm", rms_norm)
register_operator("torch", "rope", rope)
