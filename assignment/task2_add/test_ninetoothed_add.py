import pytest
import torch

from ninetoothed_add import nt_add_1d


@pytest.mark.parametrize("n", [1, 127, 1024, 100003, 1048576])
def test_ninetoothed_add(n):
    lhs = torch.randn(n, device="cuda", dtype=torch.float16)
    rhs = torch.randn_like(lhs)
    output = nt_add_1d(lhs, rhs)
    torch.testing.assert_close(output, lhs + rhs, atol=1e-2, rtol=1e-2)
