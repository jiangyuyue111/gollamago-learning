from unittest import mock

import pytest
import torch

import operators
import operators.ninetoothed_ops as ninetoothed_ops
from benchmarks.compare_results import compare


def test_torch_operators_match_references():
    input = torch.randn(2, 3, 8)
    weight = torch.randn(8)

    rms = operators.get_operator("rms_norm", "torch")

    torch.testing.assert_close(
        rms(input, weight, 1e-5),
        input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-5) * weight,
    )


def test_torch_rope_matches_half_rotation_reference():
    input = torch.randn(2, 3, 4, 8)
    sin = torch.randn(3, 4)
    cos = torch.randn(3, 4)
    actual = operators.get_operator("rope", "torch")(input, sin, cos)
    first, second = input[..., :4], input[..., 4:]
    expected = torch.cat(
        ((first * cos[None, :, None] - second * sin[None, :, None]),
         (first * sin[None, :, None] + second * cos[None, :, None])), dim=-1
    )
    torch.testing.assert_close(actual, expected)


def test_registered_operators_report_backend_implementation():
    assert operators.get_registered_operators("torch") == {
        "rms_norm": "torch",
        "rope": "torch",
    }


def test_ninetoothed_registration_marks_unimplemented_rope_as_fallback():
    assert operators.get_registered_operators("ninetoothed") == {
        "rms_norm": "ninetoothed",
        "rope": "torch_fallback",
    }


def test_ninetoothed_rms_norm_adapter_flattens_and_restores_shape():
    calls = {}

    def kernel(input, weight, eps, output, **kwargs):
        calls.update(
            input_shape=input.shape,
            weight_shape=weight.shape,
            weight=weight.clone(),
            eps=eps,
            block_size=kwargs["BLOCK_SIZE"],
        )
        output.copy_(input * weight)

    input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    weight = torch.arange(1, 5, dtype=torch.float32)

    actual = ninetoothed_ops._apply_rms_norm_kernel(kernel, input, weight, 1e-5)

    torch.testing.assert_close(actual, input * weight)
    assert calls["input_shape"] == torch.Size([6, 4])
    assert calls["weight_shape"] == torch.Size([6, 4])
    assert calls["eps"] == 1e-5
    assert calls["block_size"] == 4
    torch.testing.assert_close(calls["weight"], weight.expand(6, -1))


def test_ninetoothed_rms_norm_rejects_cpu_before_loading_kernel():
    with pytest.raises(RuntimeError, match="requires a CUDA-compatible accelerator"):
        ninetoothed_ops.rms_norm(torch.randn(2, 4), torch.ones(4), 1e-5)


def test_ninetoothed_kernel_load_reports_dependency_error():
    ninetoothed_ops._load_rms_norm_kernel.cache_clear()
    with (
        mock.patch.object(
            ninetoothed_ops.importlib,
            "import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'ninetoothed'", name="ninetoothed"
            ),
        ),
        pytest.raises(operators.OperatorUnavailableError, match="requires.*ninetoothed"),
    ):
        ninetoothed_ops._load_rms_norm_kernel()
    ninetoothed_ops._load_rms_norm_kernel.cache_clear()


def test_ninetoothed_missing_operator_does_not_fall_back():
    with pytest.raises(operators.OperatorUnavailableError, match="does not provide"):
        operators.get_operator("not_implemented", "ninetoothed")


def test_ninetoothed_rope_fallback_is_explicit():
    input = torch.randn(1, 2, 3, 8)
    sin = torch.randn(2, 4)
    cos = torch.randn(2, 4)

    with pytest.warns(RuntimeWarning, match="not implemented.*PyTorch reference"):
        actual = operators.dispatch(
            "rope", input, sin, cos, backend="ninetoothed"
        )

    expected = operators.get_operator("rope", "torch")(input, sin, cos)
    torch.testing.assert_close(actual, expected)


def test_missing_operator_reports_no_torch_implementation():
    with pytest.raises(operators.OperatorUnavailableError, match="No torch implementation"):
        operators.get_operator("not_implemented", "torch")


def test_unavailable_backend_falls_back_to_torch():
    with pytest.warns(RuntimeWarning, match="using torch"):
        implementation = operators.get_operator("rms_norm", "missing_backend")
    assert implementation is operators.get_operator("rms_norm", "torch")


def test_unavailable_ninetoothed_adapter_does_not_fall_back():
    import operators.registry as registry

    was_loaded = "ninetoothed" in registry._LOADED_BACKENDS
    registry._LOADED_BACKENDS.discard("ninetoothed")
    try:
        with (
            mock.patch.dict(
                registry._BACKEND_MODULES,
                {"ninetoothed": "operators.missing_ninetoothed_adapter"},
            ),
            mock.patch.object(
                registry.importlib,
                "import_module",
                side_effect=ImportError("missing adapter"),
            ),
            pytest.raises(operators.OperatorUnavailableError, match="Could not load"),
        ):
            operators.get_operator("rms_norm", "ninetoothed")
    finally:
        if was_loaded:
            registry._LOADED_BACKENDS.add("ninetoothed")


def test_dispatch_falls_back_when_backend_operator_fails():
    import operators.registry as registry

    original_get_operator = registry.get_operator

    def failing_operator(name, backend=None):
        if backend == "tilelang":
            return lambda *args: (_ for _ in ()).throw(RuntimeError("kernel failed"))
        return original_get_operator(name, backend)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(registry, "get_operator", failing_operator)
    with pytest.warns(RuntimeWarning, match="failed.*using torch"):
        result = operators.dispatch("rms_norm", torch.randn(1, 2, 4), torch.ones(4), 1e-5, backend="tilelang")
    monkeypatch.undo()
    assert result.shape == (1, 2, 4)


def test_ninetoothed_dispatch_does_not_hide_kernel_failure():
    import operators.registry as registry

    original_get_operator = registry.get_operator

    def failing_operator(name, backend=None):
        if backend == "ninetoothed":
            return lambda *args: (_ for _ in ()).throw(RuntimeError("kernel failed"))
        return original_get_operator(name, backend)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(registry, "get_operator", failing_operator)
    with pytest.raises(RuntimeError, match="kernel failed"):
        operators.dispatch(
            "rms_norm",
            torch.randn(1, 2, 4),
            torch.ones(4),
            1e-5,
            backend="ninetoothed",
        )
    monkeypatch.undo()


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
        "registered_operators": {"rms_norm": "torch", "rope": "torch"},
    }
    result = compare(
        {**shared, "backend": "torch", "tokens_per_second": 100},
        {
            **shared,
            "backend": "tilelang",
            "registered_operators": {
                "rms_norm": "tilelang",
                "rope": "torch_fallback",
            },
            "tokens_per_second": 180,
        },
    )

    assert result["speedup"] == 1.8
    assert result["improvement_percent"] == pytest.approx(80)
    assert result["modified_operators"] == {
        "rms_norm": {"baseline": "torch", "candidate": "tilelang"}
    }


def test_compare_rejects_different_outputs():
    baseline = {
        "target": "maca", "device_name": "d", "torch_version": "t",
        "maca_version": "m", "seed": 0, "batch_size": 1,
        "num_input_tokens_per_sequence": 1, "num_output_tokens_per_sequence": 1,
        "generated_token_ids": [[1]], "backend": "torch", "tokens_per_second": 1,
    }
    with pytest.raises(ValueError, match="token IDs differ"):
        compare(baseline, {**baseline, "generated_token_ids": [[2]]})
