"""Runtime implementation and accelerator-target selection."""

from __future__ import annotations

import dataclasses
import importlib

import torch


BACKEND_NAMES = ("torch", "tilelang", "maca_cpp", "jiuchi")
TARGET_NAMES = ("auto", "cuda", "maca")


@dataclasses.dataclass(frozen=True)
class BackendConfig:
    backend: str
    device: torch.device
    target: str | None

    @property
    def name(self) -> str:
        """Backward-compatible alias for older submissions."""
        return self.backend

    @property
    def tilelang_target(self) -> str | None:
        """Target passed to TileLang JIT kernels."""
        return self.target if self.backend == "tilelang" else None


_active_backend: BackendConfig | None = None


def _detect_hardware_target(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    return "maca" if getattr(torch.version, "maca", None) else "cuda"


def configure_backend(
    backend: str,
    device: str | torch.device,
    target: str = "auto",
) -> BackendConfig:
    """Validate and activate an operator implementation and hardware target."""
    global _active_backend

    if backend not in BACKEND_NAMES:
        raise ValueError(
            f"Unknown backend {backend!r}; choose one of: {', '.join(BACKEND_NAMES)}"
        )
    if target not in TARGET_NAMES:
        raise ValueError(
            f"Unknown target {target!r}; choose one of: {', '.join(TARGET_NAMES)}"
        )

    torch_device = torch.device(device)
    if backend != "torch" and torch_device.type != "cuda":
        raise RuntimeError(f"The {backend} backend requires a CUDA-compatible device")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected CUDA-compatible device is not available")

    detected_target = _detect_hardware_target(torch_device)
    resolved_target = detected_target if target == "auto" else target

    if resolved_target == "maca" and not getattr(torch.version, "maca", None):
        raise RuntimeError(
            "The maca target requires a MACA-enabled PyTorch build "
            "(torch.version.maca is missing)"
        )
    if backend == "maca_cpp" and resolved_target != "maca":
        raise RuntimeError("The maca_cpp backend requires --target maca")

    if backend == "tilelang":
        try:
            target_utils = importlib.import_module("tilelang.utils.target")
            resolved_target = str(target_utils.determine_target(resolved_target))
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "The tilelang backend requires a working TileLang installation"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"TileLang target {resolved_target!r} is not available"
            ) from error

    config = BackendConfig(backend, torch_device, resolved_target)
    _active_backend = config
    return config


def get_active_backend(*, default_to_torch: bool = False) -> BackendConfig:
    """Return the active configuration, optionally creating a CPU Torch default."""
    if _active_backend is None:
        if default_to_torch:
            return BackendConfig("torch", torch.device("cpu"), None)
        raise RuntimeError("No backend has been configured")
    return _active_backend


def synchronize(device: str | torch.device) -> None:
    """Wait for queued accelerator work before taking a timing sample."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
