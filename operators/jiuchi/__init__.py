"""Registration adapter for a student-provided ``jiuchi_kernels`` package."""

try:
    import jiuchi_kernels
except (ImportError, OSError) as error:
    from operators.registry import OperatorUnavailableError

    raise OperatorUnavailableError(
        "jiuchi requires an importable package named 'jiuchi_kernels'; "
        "see operators/jiuchi/README.md"
    ) from error

from operators.registry import register_operator


register_operator("jiuchi", "rms_norm", jiuchi_kernels.rms_norm)
register_operator("jiuchi", "silu_mul", jiuchi_kernels.silu_mul)
