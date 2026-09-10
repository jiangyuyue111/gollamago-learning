import pytest
import torch
from solution import tl_add_1d

@pytest.mark.parametrize("n", [1, 127, 1024, 100003, 1048576])
def test_vector_add(n):
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    out = tl_add_1d(a, b, BLOCK_N=1024)
    torch.testing.assert_close(out, a + b, atol=1e-2, rtol=1e-2)
