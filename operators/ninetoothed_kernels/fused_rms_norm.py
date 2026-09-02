"""Weighted RMSNorm adapted from ninetoothed-examples (Apache-2.0).

Source: https://github.com/InfiniTensor/ninetoothed-examples/blob/e873474d4b4de8e4fa427bf245da4a02512a68b1/ops/ninetoothed/kernels/fused_rms_norm.py
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, weight, eps, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input.tile((1, BLOCK_SIZE)),
        weight.tile((1, BLOCK_SIZE)),
        eps,
        output.tile((1, BLOCK_SIZE)),
    )


def application(input, weight, eps, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    variance = ntl.sum(input_fp32 * input_fp32) / input.shape[-1]
    output = input_fp32 * ntl.rsqrt(variance + eps) * weight  # noqa: F841


tensors = (Tensor(2), Tensor(2), Tensor(0), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)
