"""Operator registration and dispatch API used by the Llama model."""

from .registry import (
    OperatorUnavailableError,
    dispatch,
    get_operator,
    get_registered_operators,
    register_operator,
)

__all__ = [
    "OperatorUnavailableError",
    "dispatch",
    "get_operator",
    "get_registered_operators",
    "register_operator",
]
