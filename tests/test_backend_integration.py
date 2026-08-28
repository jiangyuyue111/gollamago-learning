import os

import pytest
import torch

import backends
import operators


pytestmark = pytest.mark.accelerator


def require_accelerator_tests():
    if os.environ.get("RUN_ACCELERATOR_TESTS") != "1":
        pytest.skip("set RUN_ACCELERATOR_TESTS=1 to run accelerator tests")


def test_tilelang_rms_norm_on_detected_accelerator():
    require_accelerator_tests()
    backends.configure_backend("tilelang", "cuda", "auto")
    input = torch.randn(2, 3, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)

    actual = operators.dispatch("rms_norm", input, weight, 1e-5)
    expected = (
        input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.07)


def test_maca_cpp_rms_norm_on_mxmaca():
    require_accelerator_tests()
    if not getattr(torch.version, "maca", None):
        pytest.skip("requires a MACA-enabled PyTorch build")
    backends.configure_backend("maca_cpp", "cuda", "maca")
    input = torch.randn(2, 3, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)

    actual = operators.dispatch("rms_norm", input, weight, 1e-5)
    expected = (
        input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
