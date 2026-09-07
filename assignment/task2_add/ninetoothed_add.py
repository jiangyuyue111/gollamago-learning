"""NineToothed vector-add exercise based on the official Basics tutorial."""

import ninetoothed
import torch
from ninetoothed import Tensor


BLOCK_SIZE = 1024


def arrangement(lhs, rhs, output):
    return (
        lhs.tile((BLOCK_SIZE,)),
        rhs.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def application(lhs, rhs, output):
    # TODO(student): replace this copy with tile-wise vector addition.
    output = lhs


_KERNEL = ninetoothed.make(
    arrangement,
    application,
    (Tensor(1), Tensor(1), Tensor(1)),
)


def nt_add_1d(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(lhs)
    _KERNEL(lhs, rhs, output)
    return output
