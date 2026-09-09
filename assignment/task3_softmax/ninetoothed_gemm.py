"""NineToothed GEMM exercise based on the official matrix-multiplication tutorial."""

import os

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor


AUTOTUNE = os.environ.get("NINETOOTHED_AUTOTUNE") == "1"
if AUTOTUNE:
    BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=32, upper_bound=128)
    BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=32, upper_bound=128)
    BLOCK_SIZE_K = ninetoothed.block_size(lower_bound=32, upper_bound=128)
else:
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64


def arrangement(lhs, rhs, output):
    # TODO(student): tile and align lhs, rhs, and output for blocked GEMM.
    raise NotImplementedError("Complete arrangement().")


def application(lhs, rhs, output):
    # TODO(student): accumulate ntl.dot over K tiles and write the output.
    raise NotImplementedError("Complete application().")


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
