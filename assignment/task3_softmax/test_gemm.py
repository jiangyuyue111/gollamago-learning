import pytest
import torch

from ninetoothed_gemm import nt_gemm


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (127, 96, 65)])
def test_gemm(m, n, k):
    lhs = torch.randn((m, k), device="cuda", dtype=torch.float16)
    rhs = torch.randn((k, n), device="cuda", dtype=torch.float16)
    output = nt_gemm(lhs, rhs)

    torch.testing.assert_close(output, lhs @ rhs, atol=2e-2, rtol=2e-2)
