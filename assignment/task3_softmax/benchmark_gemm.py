import statistics
import time

import torch

from ninetoothed_gemm import BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, nt_gemm


M = N = K = 1024
WARMUP = 10
REPEAT = 10
CALLS_PER_SAMPLE = 5


def median_latency_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(REPEAT):
        start = time.perf_counter()
        for _ in range(CALLS_PER_SAMPLE):
            fn()
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - start) * 1e6
        samples.append(elapsed_us / CALLS_PER_SAMPLE)
    return statistics.median(samples)


def tflops(latency_us):
    return 2 * M * N * K / (latency_us * 1e-6) / 1e12


torch.manual_seed(0)
lhs = torch.randn((M, K), device="cuda", dtype=torch.float16)
rhs = torch.randn((K, N), device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
first_run_start = time.perf_counter()
output = nt_gemm(lhs, rhs)
torch.cuda.synchronize()
first_run_ms = (time.perf_counter() - first_run_start) * 1e3

reference = torch.matmul(lhs, rhs)
torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
ninetoothed_us = median_latency_us(lambda: nt_gemm(lhs, rhs))
torch_us = median_latency_us(lambda: torch.matmul(lhs, rhs))

print(f"device={torch.cuda.get_device_name()}; shape={M}x{N}x{K}; dtype=float16")
print(
    f"block={BLOCK_SIZE_M}x{BLOCK_SIZE_N}x{BLOCK_SIZE_K}; "
    f"warmup={WARMUP}; repeat={REPEAT}; calls_per_sample={CALLS_PER_SAMPLE}; correct=PASS"
)
print(f"first_run={first_run_ms:.2f} ms")
print(f"NineToothed: {ninetoothed_us:.2f} us, {tflops(ninetoothed_us):.3f} TFLOPS")
print(f"PyTorch:     {torch_us:.2f} us, {tflops(torch_us):.3f} TFLOPS")
print(f"speedup_vs_PyTorch={torch_us / ninetoothed_us:.3f}x")
