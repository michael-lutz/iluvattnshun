"""Type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Hashable, TypeVar, Union

TensorTree = Any
"""Jax PyTree-like structure for tensors. Inclusive of tensors themselves."""

T = TypeVar("T", bound=Hashable)
"""Type variable for hashable types."""
