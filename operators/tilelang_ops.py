"""TileLang RMSNorm and fused SiLU-multiply examples."""

import functools

import torch

import backends
from .registry import register_operator


_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _tilelang_dtype(dtype: torch.dtype) -> str:
    try:
        return _DTYPES[dtype]
    except KeyError as error:
        raise TypeError(f"TileLang examples do not support dtype {dtype}") from error


def _target() -> str:
    config = backends.get_active_backend()
    if config.backend != "tilelang" or config.target is None:
        raise RuntimeError("TileLang operator called without an active TileLang backend")
    return config.target


@functools.lru_cache(maxsize=None)
def _compile_silu_mul(dtype: str, target: str):
    import tilelang
    import tilelang.language as T

    length = T.dynamic("length")
    block = 256

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            gate: T.Tensor((length,), dtype),
            up: T.Tensor((length,), dtype),
            output: T.Tensor((length,), dtype),
        ):
            with T.Kernel(T.ceildiv(length, block), threads=block) as block_id:
                for offset in T.Parallel(block):
                    index = block_id * block + offset
                    if index < length:
                        output[index] = gate[index] * T.sigmoid(gate[index]) * up[index]

        return main

    return kernel()


@functools.lru_cache(maxsize=None)
def _compile_rms_norm(columns: int, eps: float, dtype: str, target: str):
    import tilelang
    import tilelang.language as T

    rows = T.dynamic("rows")
    block_columns = 512

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            input: T.Tensor((rows, columns), dtype),
            weight: T.Tensor((columns,), dtype),
            output: T.Tensor((rows, columns), dtype),
        ):
            with T.Kernel(rows, threads=128) as row:
                input_shared = T.alloc_shared((block_columns,), dtype)
                square_fragment = T.alloc_fragment((block_columns,), T.float32)
                square_sum = T.alloc_fragment((1,), T.float32)

                T.clear(square_fragment)
                for chunk in range(T.ceildiv(columns, block_columns)):
                    T.copy(input[row, chunk * block_columns], input_shared)
                    for offset in T.Parallel(block_columns):
                        square_fragment[offset] += (
                            input_shared[offset] * input_shared[offset]
                        )

                T.reduce_sum(square_fragment, square_sum, dim=0)
                square_sum[0] = T.rsqrt(square_sum[0] / columns + eps)

                for chunk in range(T.ceildiv(columns, block_columns)):
                    for offset in T.Parallel(block_columns):
                        index = chunk * block_columns + offset
                        if index < columns:
                            output[row, index] = (
                                input[row, index] * square_sum[0] * weight[index]
                            )

        return main

    return kernel()


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up must have identical shapes")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("TileLang silu_mul expects contiguous tensors")
    kernel = _compile_silu_mul(_tilelang_dtype(gate.dtype), _target())
    return kernel(gate.reshape(-1), up.reshape(-1)).reshape(gate.shape)


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if input.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight size must match the final input dimension")
    if not input.is_contiguous() or not weight.is_contiguous():
        raise ValueError("TileLang rms_norm expects contiguous tensors")
    columns = input.shape[-1]
    kernel = _compile_rms_norm(columns, eps, _tilelang_dtype(input.dtype), _target())
    return kernel(input.reshape(-1, columns), weight).reshape(input.shape)


register_operator("tilelang", "rms_norm", rms_norm)
register_operator("tilelang", "silu_mul", silu_mul)
