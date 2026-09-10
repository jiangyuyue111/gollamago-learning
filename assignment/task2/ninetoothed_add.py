"""NineToothed vector-add exercise based on the official Basics tutorial."""

import os

import ninetoothed
import torch
from ninetoothed import Tensor


AUTOTUNE = os.environ.get("NINETOOTHED_AUTOTUNE") == "1"
BLOCK_SIZE = (
    ninetoothed.block_size(lower_bound=256, upper_bound=1024)
    if AUTOTUNE
    else 1024
)


def arrangement(lhs, rhs, output):
    # TODO(student): tile lhs, rhs, and output with BLOCK_SIZE.
    raise NotImplementedError("Complete arrangement().")


def application(lhs, rhs, output):
    # TODO(student): compute the tile-wise vector addition.
    raise NotImplementedError("Complete application().")


_KERNEL = ninetoothed.make(
    arrangement,
    application,
    (Tensor(1), Tensor(1), Tensor(1)),
)


def nt_add_1d(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(lhs)
    _KERNEL(lhs, rhs, output)
    return output
