import torch

from ninetoothed_gemm import nt_gemm


def test_ninetoothed_gemm():
    torch.manual_seed(0)

    shape = (512, 512)
    lhs = torch.randn(shape, device="cuda", dtype=torch.float16)
    rhs = torch.randn(shape, device="cuda", dtype=torch.float16)

    output = nt_gemm(lhs, rhs)
    reference = torch.mm(lhs, rhs)

    assert torch.allclose(output, reference, atol=0.025, rtol=0.025)
