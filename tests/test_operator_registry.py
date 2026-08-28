import pytest
import torch

import operators
from benchmarks.compare_results import compare


def test_torch_operators_match_references():
    input = torch.randn(2, 3, 8)
    weight = torch.randn(8)
    gate = torch.randn(2, 3, 8)
    up = torch.randn_like(gate)

    rms = operators.get_operator("rms_norm", "torch")
    silu_mul = operators.get_operator("silu_mul", "torch")

    torch.testing.assert_close(
        rms(input, weight, 1e-5),
        input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-5) * weight,
    )
    torch.testing.assert_close(silu_mul(gate, up), torch.nn.functional.silu(gate) * up)


def test_missing_operator_does_not_fall_back_to_torch():
    with pytest.raises(operators.OperatorUnavailableError, match="automatic fallback"):
        operators.get_operator("not_implemented", "torch")


def test_compare_uses_throughput_ratio():
    shared = {
        "target": "maca",
        "device_name": "test-device",
        "torch_version": "test",
        "maca_version": "test",
        "seed": 0,
        "batch_size": 2,
        "num_input_tokens_per_sequence": 8,
        "num_output_tokens_per_sequence": 4,
        "generated_token_ids": [[1, 2], [3, 4]],
    }
    result = compare(
        {**shared, "backend": "torch", "tokens_per_second": 100},
        {**shared, "backend": "tilelang", "tokens_per_second": 180},
    )

    assert result["speedup"] == 1.8
    assert result["improvement_percent"] == pytest.approx(80)


def test_compare_rejects_different_outputs():
    baseline = {
        "target": "maca", "device_name": "d", "torch_version": "t",
        "maca_version": "m", "seed": 0, "batch_size": 1,
        "num_input_tokens_per_sequence": 1, "num_output_tokens_per_sequence": 1,
        "generated_token_ids": [[1]], "backend": "torch", "tokens_per_second": 1,
    }
    with pytest.raises(ValueError, match="token IDs differ"):
        compare(baseline, {**baseline, "generated_token_ids": [[2]]})
