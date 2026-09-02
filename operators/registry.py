"""Explicit registry for interchangeable operator implementations."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable

import backends


Operator = Callable[..., object]
_BACKEND_MODULES = {
    "torch": "operators.torch_ops",
    "tilelang": "operators.tilelang_ops",
    "maca_cpp": "operators.maca_cpp",
    "ninetoothed": "operators.ninetoothed_ops",
}
_OPERATORS: dict[tuple[str, str], Operator] = {}
_TORCH_FALLBACK_OPERATORS: set[tuple[str, str]] = set()
_LOADED_BACKENDS: set[str] = set()


class OperatorUnavailableError(RuntimeError):
    """Raised when a backend does not provide a requested operator."""


def register_operator(
    backend: str,
    name: str,
    implementation: Operator,
    *,
    fallback_to_torch: bool = False,
) -> None:
    key = (backend, name)
    if key in _OPERATORS:
        raise ValueError(f"Operator {name!r} is already registered for {backend!r}")
    _OPERATORS[key] = implementation
    if fallback_to_torch:
        _TORCH_FALLBACK_OPERATORS.add(key)


def get_registered_operators(backend: str | None = None) -> dict[str, str]:
    """Return registered operators and whether each uses this backend or Torch."""
    selected = backend or backends.get_active_backend(default_to_torch=True).backend
    _load_backend(selected)
    return {
        name: "torch_fallback" if (selected, name) in _TORCH_FALLBACK_OPERATORS else selected
        for registered_backend, name in sorted(_OPERATORS)
        if registered_backend == selected
    }


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
    try:
        _load_backend(selected)
    except OperatorUnavailableError as error:
        warnings.warn(f"Backend {selected!r} is unavailable ({error}); using torch", RuntimeWarning)
        selected = "torch"
        _load_backend(selected)
    try:
        return _OPERATORS[(selected, name)]
    except KeyError as error:
        if selected != "torch":
            warnings.warn(
                f"Backend {selected!r} does not provide {name!r}; using torch",
                RuntimeWarning,
            )
            return get_operator(name, "torch")
        raise OperatorUnavailableError(
            f"No torch implementation is registered for operator {name!r}"
        ) from error


def dispatch(name: str, *args, backend: str | None = None, **kwargs):
    selected = backend or backends.get_active_backend(default_to_torch=True).backend
    implementation = get_operator(name, selected)
    try:
        return implementation(*args, **kwargs)
    except Exception as error:
        if selected == "torch":
            raise
        warnings.warn(
            f"Backend {selected!r} failed for {name!r} ({error}); using torch",
            RuntimeWarning,
        )
        return get_operator(name, "torch")(*args, **kwargs)
