"""Type definitions."""

from __future__ import annotations

from typing import Union

import torch

# Jax PyTree-like structure for tensors. Inclusive of tensors themselves.
TensorTree = Union[torch.Tensor, list["TensorTree"], tuple["TensorTree", ...], dict[str, "TensorTree"]]
