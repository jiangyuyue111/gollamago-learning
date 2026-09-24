import time

import torch

from solution import tl_add_1d


SIZES = [1, 127, 1024, 100003, 1048576]
BLOCK_SIZES = [256, 1024]


def timed(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(repeat):
        fn()

    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat


print("N\tBLOCK_N\tTileLang(us)\tGB/s\tmax_abs_error")

for n in SIZES:
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    ref = a + b

    for block_n in BLOCK_SIZES:
        kernel = tl_add_1d.compile(N=n, BLOCK_N=block_n)
        out = kernel(a, b)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

        elapsed_us = timed(lambda: kernel(a, b))
        effective_gbps = n * 6 / (elapsed_us * 1e-6) / 1e9
        max_error = (out - ref).abs().max().item()

        print(
            f"{n}\t{block_n}\t{elapsed_us:.2f}\t\t"
            f"{effective_gbps:.2f}\t{max_error:.3e}"
        )
