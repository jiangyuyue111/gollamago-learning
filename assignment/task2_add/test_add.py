import pytest
import torch

from ninetoothed_add import nt_add_1d
from solution import tl_add_1d


@pytest.mark.parametrize("n", [1, 127, 1024, 100003, 1048576])
def test_vector_add(n):
    lhs = torch.randn(n, device="cuda", dtype=torch.float16)
    rhs = torch.randn_like(lhs)
    reference = lhs + rhs

    ninetoothed_output = nt_add_1d(lhs, rhs)
    tilelang_output = tl_add_1d(lhs, rhs, BLOCK_N=1024)

    torch.testing.assert_close(ninetoothed_output, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tilelang_output, reference, atol=1e-2, rtol=1e-2)
