"""Base trainer class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import Generic, Iterable, Literal, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim

from iluvattnshun.types import TensorTree


class Logger:
    """Creates beautiful console logs & interfaces with tensorboard."""

    def __init__(self, precision: int = 4, log_every_n_seconds: float = 30.0):
        """Initialize the logger.

        Args:
            precision: Number of decimal places to show for float values.
            log_every_n_seconds: Log every n seconds.
        """
        self.precision = precision
        self.log_every_n_seconds = log_every_n_seconds
        self.start_time = time()
        self.step = 0
        self.last_log_time = time()

    def _format_number(self, value: float | int) -> str:
        """Format a number with appropriate precision."""
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.{self.precision}f}"

    def _format_time(self, seconds: float) -> str:
        """Format elapsed time in a human readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def log(self, metrics: dict[str, float], mode: Literal["train", "val"]) -> None:
        """Log metrics to the console window.

        Args:
            metrics: Dictionary of metric names to values.
            mode: Whether this is training or validation metrics.
        """
        if time() - self.last_log_time > self.log_every_n_seconds:
            self.last_log_time = time()
            # TODO: do nice windowing and ascii art
            mode_str = "🟢 TRAIN" if mode == "train" else "🟡 VAL"
            print(f"\n{mode_str} | Step: {self.step:,} | Time: {self._format_time(time() - self.start_time)}")
            print("─" * 50)
            for name, value in metrics.items():
                print(f"- {name}: {self._format_number(value)}")
            print("─" * 50)


@dataclass
class TrainerConfig:
    """Trainer configuration."""

    # Training outer loop
    num_epochs: int
    batch_size: int

    # Logging
    log_every_n_seconds: float
    log_fp: int = 4


ConfigType = TypeVar("ConfigType", bound=TrainerConfig)


class Trainer(ABC, Generic[ConfigType]):
    """Base trainer class."""

    def __init__(self, config: ConfigType):
        """Initialize the trainer."""
        self.config = config
        self.logger = Logger(precision=config.log_fp, log_every_n_seconds=config.log_every_n_seconds)
        self.step = 0

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Get the model."""

    @abstractmethod
    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get the optimizer."""

    @abstractmethod
    def get_loss(self, model: nn.Module, batch: TensorTree) -> torch.Tensor:
        """Get loss for a batch."""

    @abstractmethod
    def get_train_dataloader(self) -> Iterable[TensorTree]:
        """Get the train dataloader."""

    @abstractmethod
    def get_val_dataloader(self) -> Iterable[TensorTree]:
        """Get the val dataloader."""

    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, batch: TensorTree) -> dict[str, float]:
        """Train step."""
        # TODO: think about ownership of .train and .zero_grad for safe override
        assert model.training, "Model must be in training mode"
        optimizer.zero_grad()
        loss = self.get_loss(model, batch)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def val_step(self, model: nn.Module, batch: TensorTree) -> dict[str, float]:
        """Returns eval metrics."""
        assert not model.training, "Model must be in evaluation mode"
        loss = self.get_loss(model, batch)
        return {"loss": loss.item()}

    def run(self) -> None:
        """Creates or loads training variables and begins training."""
        model = self.get_model()
        optimizer = self.get_optimizer(model)
        train_iterator = iter(self.get_train_dataloader())
        val_iterator = iter(self.get_val_dataloader())

        while True:
            model.train()
            batch = next(train_iterator)
            metrics = self.train_step(model, optimizer, batch)
            self.logger.log(metrics, mode="train")

            model.eval()
            batch = next(val_iterator)
            metrics = self.val_step(model, batch)
            self.logger.log(metrics, mode="val")

            self.step += 1
