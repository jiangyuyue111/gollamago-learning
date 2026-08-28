"""Reference PyTorch operators used for correctness and baseline measurements."""

import torch

from .registry import register_operator


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return input * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


register_operator("torch", "rms_norm", rms_norm)
