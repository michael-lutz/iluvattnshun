"""Base trainer class."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from time import time
from typing import Generic, Iterable, Iterator, Literal, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from iluvattnshun.logger import Logger
from iluvattnshun.types import TensorTree


@dataclass(kw_only=True)
class TrainerConfig:
    """Trainer configuration."""

    # Training outer loop
    num_epochs: int
    """Number of training epochs. If -1, training will continue indefinitely."""
    batch_size: int
    """Batch size."""

    # Logging
    log_every_n_seconds: float = 30.0
    """Log every n seconds."""
    log_fp: int = 4
    """Log float precision."""


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

    def validation_metrics(self, model: nn.Module, batch: TensorTree) -> dict[str, float | str]:
        """(Optional) Get additional validation metrics for a batch."""
        return {}

    @abstractmethod
    def get_train_dataloader(self) -> Iterable[TensorTree]:
        """Get the train dataloader."""

    @abstractmethod
    def get_val_dataloader(self) -> Iterable[TensorTree]:
        """Get the val dataloader."""

    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, batch: TensorTree) -> dict[str, float | str]:
        """Train step."""
        # TODO: think about ownership of .train and .zero_grad for safe override
        assert model.training, "Model must be in training mode"
        optimizer.zero_grad()
        loss = self.get_loss(model, batch)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def val_step(self, model: nn.Module, batch: TensorTree) -> dict[str, float | str]:
        """Returns eval metrics."""
        assert not model.training, "Model must be in evaluation mode"
        loss = self.get_loss(model, batch)
        metrics: dict[str, float | str] = {"loss": loss.item()}
        metrics.update(self.validation_metrics(model, batch))
        return metrics

    def run(self) -> None:
        """Creates or loads training variables and begins training."""
        model = self.get_model()
        optimizer = self.get_optimizer(model)
        train_loader = self.get_train_dataloader()
        val_loader = self.get_val_dataloader()

        def inner_train_loop(epoch: int) -> None:
            # Clasic batched training loop
            model.train()
            for batch in train_loader:
                metrics = self.train_step(model, optimizer, batch)
                self.logger.log({"epoch": epoch, **metrics}, mode="train")

            # Running eval every epoch and combining metrics over all batches
            model.eval()
            eval_metrics: dict[str, float | str] = {}
            eval_size = 0
            for batch in val_loader:
                metrics = self.val_step(model, batch)
                eval_metrics = {
                    k: eval_metrics.get(k, 0) + v if isinstance(v, float) else v for k, v in metrics.items()
                }
                eval_size += 1

            eval_metrics = {k: v / eval_size if isinstance(v, float) else v for k, v in eval_metrics.items()}
            self.logger.log({"epoch": epoch, **eval_metrics}, mode="val")

            self.step += 1

        if self.config.num_epochs == -1:
            while True:
                inner_train_loop(-1)
        else:
            for epoch in range(self.config.num_epochs):
                inner_train_loop(epoch)
