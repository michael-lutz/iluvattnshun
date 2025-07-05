"""Utility functions."""

import os
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Literal

import torch
import torch.nn as nn
import yaml

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


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: dict[Literal["train", "val"], int],
    metrics: dict[str, float | str],
    scheduler: Any = None,
    **kwargs: Any,
) -> None:
    """Save a checkpoint of the model and training state.

    Args:
        path: Path to save the checkpoint to
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch: Current epoch number
        step: Current step number
        metrics: Dictionary of metrics to save
        **kwargs: Additional items to save in the checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        **kwargs,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | None = None,
) -> tuple[int, dict[Literal["train", "val"], int], dict[str, float | str]]:
    """Load the model and opt in-place and returns the epoch, step, and metrics.

    Args:
        path: Path to load the checkpoint from
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        map_location: Device to map the checkpoint to (e.g. 'cuda:0' or 'cpu')

    Returns:
        Tuple of (epoch, step, metrics) from the checkpoint
    """
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch"), checkpoint.get("step"), checkpoint.get("metrics")


def update_config_from_cli(config: Any) -> Any:
    """Update a dataclass instance with matching CLI args (e.g., --param value)."""
    args = sys.argv[1:]
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=")
            key = key.lstrip("--")
            value = type(getattr(config, key))(value)
            config = replace(config, **{key: value})
        elif "--" in arg:
            key = arg.lstrip("--")
            config = replace(config, **{key: True})
        else:
            raise ValueError(f"Invalid argument: {arg}. Expected --param_name=value or --bool_param_name")

    return config


def get_yaml_string(config: Any) -> str:
    """Get a YAML string from a dataclass instance."""
    config_dict = asdict(config)
    return_str: str = yaml.dump(config_dict)
    return return_str


def load_config_from_yaml(yaml_path: str, cls: type[Any]) -> Any:
    """Load a config from a YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return cls(**config)
