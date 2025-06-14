"""Utility functions."""

import torch

from iluvattnshun.types import TensorTree


def move_to_device(batch: TensorTree, device: str) -> TensorTree:
    """Move a batch of data to a specific device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        res = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.to(device)
            else:
                res[k] = move_to_device(v, device)
        return res
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch
