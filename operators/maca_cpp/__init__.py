"""Registration adapter for the bundled native MXMACA extension."""

try:
    from . import maca_kernels
except (ImportError, OSError) as error:
    from operators.registry import OperatorUnavailableError

    raise OperatorUnavailableError(
        "maca_cpp requires the bundled 'maca_kernels' extension to be built; "
        "see operators/maca_cpp/README.md"
    ) from error

from operators.registry import register_operator


register_operator("maca_cpp", "rms_norm", maca_kernels.rms_norm)
