"""NineToothed GEMM exercise based on the official matrix-multiplication tutorial."""

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor


BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64


def arrangement(lhs, rhs, output):
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    # TODO(student): iterate over K tiles and accumulate ntl.dot(lhs[k], rhs[k]).

    output = accumulator.to(ntl.float16)


_KERNEL = ninetoothed.make(
    arrangement,
    application,
    (Tensor(2), Tensor(2), Tensor(2)),
)


def nt_gemm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]),
        device=lhs.device,
        dtype=torch.float16,
    )
    _KERNEL(lhs, rhs, output)
    return output
