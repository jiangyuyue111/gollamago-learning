import torch

from ninetoothed_add import nt_add_1d


def test_ninetoothed_add():
    torch.manual_seed(0)

    size = 98432
    lhs = torch.randn(size, device="cuda", dtype=torch.float16)
    rhs = torch.randn_like(lhs)

    output = nt_add_1d(lhs, rhs)
    reference = torch.add(lhs, rhs)

    assert torch.allclose(output, reference)
