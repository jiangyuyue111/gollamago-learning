"""Registration adapter for the bundled native MXMACA extension."""

import warnings

try:
    from . import maca_kernels
except (ImportError, OSError) as error:
    from operators.registry import OperatorUnavailableError

    raise OperatorUnavailableError(
        "maca_cpp requires the bundled 'maca_kernels' extension to be built; "
        "see operators/INTEGRATION.md"
    ) from error

from operators.registry import register_operator


register_operator("maca_cpp", "rms_norm", maca_kernels.rms_norm)


def _torch_rope_fallback(*args, **kwargs):
    """Temporary reference path until a student adds the native RoPE kernel."""
    warnings.warn(
        "MXMACA rope is not implemented; using the PyTorch reference. "
        "Add src/rope.maca and expose maca_kernels.rope for performance work.",
        RuntimeWarning,
        stacklevel=2,
    )
    from operators.torch_ops import rope as torch_rope

    return torch_rope(*args, **kwargs)


# Once a student exposes maca_kernels.rope, it is picked up automatically. Until then,
# the already-connected model path remains runnable through the Torch reference.
register_operator(
    "maca_cpp",
    "rope",
    getattr(maca_kernels, "rope", _torch_rope_fallback),
    fallback_to_torch=not hasattr(maca_kernels, "rope"),
)
