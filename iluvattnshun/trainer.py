"""Base trainer class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim

from iluvattnshun.logger import Logger
from iluvattnshun.types import TensorTree
from iluvattnshun.utils import move_to_device


@dataclass(kw_only=True)
class TrainerConfig:
    """Trainer configuration."""

    # Training outer loop
    num_epochs: int
    """Number of training epochs. If -1, training will continue indefinitely."""
    batch_size: int
    """Batch size."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for training."""
    eval_every_n_samples: int
    """Evaluate every n samples."""

    # Logging
    log_every_n_seconds: float = 30.0
    """Log every n seconds."""
    log_fp: int = 4
    """Log float precision."""
    tensorboard_logdir: str | None = None
    """Tensorboard log directory."""


ConfigType = TypeVar("ConfigType", bound=TrainerConfig)


class Trainer(ABC, Generic[ConfigType]):
    """Base trainer class."""

    def __init__(self, config: ConfigType):
        """Initialize the trainer."""
        self.config = config
        self.logger = Logger(
            precision=config.log_fp,
            log_every_n_seconds=config.log_every_n_seconds,
            tensorboard_logdir=config.tensorboard_logdir,
        )
        self.init_state()

    def init_state(self) -> None:
        """(Optional) Initialize any state (happens at end of __init__)."""
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Get the model."""

    @abstractmethod
    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get the optimizer."""

    @abstractmethod
    def get_loss(self, model: nn.Module, batch: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Get loss and predictions for a batch."""

    def val_metrics(self, model: nn.Module, batch: TensorTree, preds: torch.Tensor) -> dict[str, float | str]:
        """(Optional) Get additional validation metrics for a batch."""
        return {}

    def post_val_metrics(self, model: nn.Module) -> dict[str, float | str]:
        """(Optional) Metrics unrelated to data (e.g. sample generations)."""
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
        loss, _ = self.get_loss(model, batch)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def val_step(self, model: nn.Module, batch: TensorTree) -> dict[str, float | str]:
        """Returns eval metrics."""
        assert not model.training, "Model must be in evaluation mode"
        loss, preds = self.get_loss(model, batch)
        metrics: dict[str, float | str] = {"loss": loss.item()}
        metrics.update(self.val_metrics(model, batch, preds))
        return metrics

    def run(self) -> None:
        """Creates or loads training variables and begins training."""
        model = self.get_model().to(self.config.device)
        optimizer = self.get_optimizer(model)
        train_loader = self.get_train_dataloader()
        val_loader = self.get_val_dataloader()

        epoch_dec = self.config.num_epochs
        training_samples = 0
        eval_steps = 0
        while epoch_dec != 0:
            epoch = self.config.num_epochs - epoch_dec
            for batch in train_loader:
                # run full evaluation every n samples
                model.eval()
                if training_samples >= eval_steps * self.config.eval_every_n_samples:
                    eval_metrics: dict[str, float | str] = {}
                    eval_size = 0

                    for batch in val_loader:
                        batch = move_to_device(batch, self.config.device)
                        metrics = self.val_step(model, batch)

                        # handle float and string metrics separately
                        for k, v in metrics.items():
                            if isinstance(v, float):
                                if k not in eval_metrics:
                                    eval_metrics[k] = 0.0
                                prev = eval_metrics[k]
                                assert isinstance(prev, float)
                                eval_metrics[k] = prev + v
                            else:
                                eval_metrics[k] = v  # only keep the last string
                        eval_size += 1

                    # average the float metrics
                    eval_metrics = {k: v / eval_size if isinstance(v, float) else v for k, v in eval_metrics.items()}
                    eval_metrics.update(self.post_val_metrics(model))
                    self.logger.log(
                        eval_metrics,
                        mode="val",
                        header={"epoch": epoch, "samples": training_samples},
                    )
                    eval_steps += 1
                # classic train step
                model.train()
                training_samples += self.config.batch_size
                batch = move_to_device(batch, self.config.device)
                metrics = self.train_step(model, optimizer, batch)
                self.logger.log(
                    metrics,
                    mode="train",
                    header={"epoch": epoch, "samples": training_samples},
                )

            epoch_dec -= 1
