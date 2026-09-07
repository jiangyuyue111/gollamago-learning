import time

import torch

from ninetoothed_add import nt_add_1d


SIZES = [1, 127, 1024, 100003, 1048576]


def timed(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat


print("N\tcorrect\tNineToothed(us)\tPyTorch(us)\tmax_abs_error")
for n in SIZES:
    lhs = torch.randn(n, device="cuda", dtype=torch.float16)
    rhs = torch.randn_like(lhs)
    output = nt_add_1d(lhs, rhs)
    reference = lhs + rhs
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    print(
        f"{n}\tPASS\t{timed(lambda: nt_add_1d(lhs, rhs)):.2f}\t\t"
        f"{timed(lambda: lhs + rhs):.2f}\t\t"
        f"{(output - reference).abs().max().item():.3e}"
    )
