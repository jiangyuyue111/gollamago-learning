import torch
import triton

from ninetoothed_add import nt_add_1d


SIZES = [1, 127, 1024, 100003, 1048576]

print("N\tcorrect\tNineToothed(ms)\tPyTorch(ms)\tmax_abs_error")
for n in SIZES:
    lhs = torch.randn(n, device="cuda", dtype=torch.float16)
    rhs = torch.randn_like(lhs)
    output = nt_add_1d(lhs, rhs)
    reference = lhs + rhs
    assert torch.allclose(output, reference, atol=1e-2, rtol=1e-2)
    print(
        f"{n}\tPASS\t{triton.testing.do_bench(lambda: nt_add_1d(lhs, rhs)):.4f}\t\t"
        f"{triton.testing.do_bench(lambda: lhs + rhs):.4f}\t\t"
        f"{(output - reference).abs().max().item():.3e}"
    )
