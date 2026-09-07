import statistics
import time

import torch

from ninetoothed_add import BLOCK_SIZE as NINETOOTHED_BLOCK_SIZE
from ninetoothed_add import nt_add_1d
from solution import tl_add_1d


SIZES = [1, 127, 1024, 100003, 1048576]
TILELANG_BLOCK_SIZES = [256, 1024]
WARMUP = 10
REPEAT = 20
CALLS_PER_SAMPLE = 50


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


torch.manual_seed(0)
print(
    f"NineToothed BLOCK_SIZE={NINETOOTHED_BLOCK_SIZE}; "
    f"TileLang BLOCK_N={TILELANG_BLOCK_SIZES}"
)
print(
    f"device={torch.cuda.get_device_name()}; dtype=float16; "
    f"warmup={WARMUP}; repeat={REPEAT}; calls_per_sample={CALLS_PER_SAMPLE}"
)
print("N\tTL_BLOCK\tcorrect\tNineToothed(us)\tTileLang(us)\tPyTorch(us)\tNT/PT\tTL/PT")

for n in SIZES:
    lhs = torch.randn(n, device="cuda", dtype=torch.float16)
    rhs = torch.randn_like(lhs)
    reference = lhs + rhs

    ninetoothed_output = nt_add_1d(lhs, rhs)
    torch.testing.assert_close(ninetoothed_output, reference, atol=1e-2, rtol=1e-2)
    ninetoothed_us = median_latency_us(lambda: nt_add_1d(lhs, rhs))
    torch_us = median_latency_us(lambda: torch.add(lhs, rhs))

    for block_size in TILELANG_BLOCK_SIZES:
        tilelang_kernel = tl_add_1d.compile(N=n, BLOCK_N=block_size)
        tilelang_output = tilelang_kernel(lhs, rhs)
        torch.testing.assert_close(tilelang_output, reference, atol=1e-2, rtol=1e-2)
        tilelang_us = median_latency_us(lambda: tilelang_kernel(lhs, rhs))

        print(
            f"{n}\t{block_size}\t\tPASS\t{ninetoothed_us:.2f}\t\t"
            f"{tilelang_us:.2f}\t\t{torch_us:.2f}\t\t"
            f"{torch_us / ninetoothed_us:.3f}x\t{torch_us / tilelang_us:.3f}x"
        )
