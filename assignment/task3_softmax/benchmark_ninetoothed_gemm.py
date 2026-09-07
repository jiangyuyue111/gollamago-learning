import time

import torch

from ninetoothed_gemm import BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, nt_gemm


M = N = K = 1024


def timed(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat


def tflops(latency_us):
    return 2 * M * N * K / (latency_us * 1e-6) / 1e12


lhs = torch.randn((M, K), device="cuda", dtype=torch.float16)
rhs = torch.randn((K, N), device="cuda", dtype=torch.float16)
output = nt_gemm(lhs, rhs)
reference = torch.matmul(lhs, rhs)
torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)

ninetoothed_us = timed(lambda: nt_gemm(lhs, rhs))
torch_us = timed(lambda: torch.matmul(lhs, rhs))
print(f"block={BLOCK_SIZE_M}x{BLOCK_SIZE_N}x{BLOCK_SIZE_K}; correct=PASS")
print(f"NineToothed: {ninetoothed_us:.2f} us, {tflops(ninetoothed_us):.3f} TFLOPS")
print(f"PyTorch:     {torch_us:.2f} us, {tflops(torch_us):.3f} TFLOPS")
print(f"speedup_vs_PyTorch={torch_us / ninetoothed_us:.3f}x")
