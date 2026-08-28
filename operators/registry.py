"""Explicit registry for interchangeable operator implementations."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import backends


Operator = Callable[..., object]
_BACKEND_MODULES = {
    "torch": "operators.torch_ops",
    "tilelang": "operators.tilelang_ops",
    "maca_cpp": "operators.maca_cpp",
    "jiuchi": "operators.jiuchi",
}
_OPERATORS: dict[tuple[str, str], Operator] = {}
_LOADED_BACKENDS: set[str] = set()


class OperatorUnavailableError(RuntimeError):
    """Raised when a backend does not provide a requested operator."""


def register_operator(backend: str, name: str, implementation: Operator) -> None:
    key = (backend, name)
    if key in _OPERATORS:
        raise ValueError(f"Operator {name!r} is already registered for {backend!r}")
    _OPERATORS[key] = implementation


def _load_backend(backend: str) -> None:
    if backend in _LOADED_BACKENDS:
        return
    module = _BACKEND_MODULES.get(backend)
    if module is None:
        raise OperatorUnavailableError(f"Unknown operator backend {backend!r}")
    try:
        importlib.import_module(module)
    except OperatorUnavailableError:
        raise
    except (ImportError, OSError) as error:
        raise OperatorUnavailableError(
            f"Could not load the {backend!r} operator backend: {error}"
        ) from error
    _LOADED_BACKENDS.add(backend)


def get_operator(name: str, backend: str | None = None) -> Operator:
    selected = backend or backends.get_active_backend(default_to_torch=True).backend
    _load_backend(selected)
    try:
        return _OPERATORS[(selected, name)]
    except KeyError as error:
        raise OperatorUnavailableError(
            f"Backend {selected!r} does not provide operator {name!r}; "
            "automatic fallback to PyTorch is disabled"
        ) from error


def dispatch(name: str, *args, backend: str | None = None, **kwargs):
    return get_operator(name, backend)(*args, **kwargs)
