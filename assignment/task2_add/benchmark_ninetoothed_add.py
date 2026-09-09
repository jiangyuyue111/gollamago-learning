import torch
import triton

from ninetoothed_add import nt_add_1d


SIZES = [2**i for i in range(18, 28)]


if __name__ == "__main__":
    torch.manual_seed(0)
    print("size\tcorrect\tNineToothed(ms)\tPyTorch(ms)")

    for size in SIZES:
        lhs = torch.randn(size, device="cuda", dtype=torch.float16)
        rhs = torch.randn_like(lhs)

        output = nt_add_1d(lhs, rhs)
        reference = torch.add(lhs, rhs)
        assert torch.allclose(output, reference)

        ninetoothed_ms = triton.testing.do_bench(lambda: nt_add_1d(lhs, rhs))
        torch_ms = triton.testing.do_bench(lambda: torch.add(lhs, rhs))
        print(f"{size}\tPASS\t{ninetoothed_ms:.4f}\t\t{torch_ms:.4f}")
