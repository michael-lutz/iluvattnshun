"""Base trainer class."""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import Any, Generic, Iterable, Literal, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim

from iluvattnshun.logger import Logger
from iluvattnshun.types import TensorTree
from iluvattnshun.utils import (
    get_yaml_string,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    update_config_from_cli,
)


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
    grad_norm_max: float = 1.0
    """Maximum gradient norm."""

    # Logging
    log_every_n_seconds: float = 30.0
    """Log every n seconds."""
    log_fp: int = 4
    """Log float precision."""
    tensorboard_logdir: str
    """Tensorboard log directory."""
    run_name: str = ""
    """Name of the run. If empty, uses timestamp."""

    # Checkpointing
    save_every_n_seconds: int = 1
    """Save checkpoint every n epochs."""
    overwrite_existing_checkpoints: bool = False
    """Deletes existing checkpoints when saving."""
    save_model_path: str | None = None
    """Path to save checkpoints to. If None, saves to tensorboard_logdir/checkpoints."""
    load_model_path: str | None = None
    """Path to .pt file to load from. If None, starts from scratch."""


ConfigType = TypeVar("ConfigType", bound=TrainerConfig)


class Trainer(ABC, Generic[ConfigType]):
    """Base trainer class."""

    def __init__(self, config: ConfigType):
        """Initialize the trainer."""
        config = update_config_from_cli(config)
        self.config = config
        self.logger = Logger(
            tensorboard_logdir=config.tensorboard_logdir,
            precision=config.log_fp,
            log_every_n_seconds=config.log_every_n_seconds,
            name=config.run_name or None,
        )

        # log the config
        config_text = get_yaml_string(config)
        self.logger.log_text(self.logger.run_name + ".yaml", config_text)

        # log the script
        main_module = sys.modules["__main__"]
        if hasattr(main_module, "__file__"):
            main_file = main_module.__file__
            if main_file is not None:
                script_name = self.logger.run_name + ".py"
                script_text = open(main_file, "r").read()
                self.logger.log_text(script_name, script_text, save_to_file=True)

        # setup checkpoint directory
        self.save_model_path = self.config.save_model_path or self.logger.log_dir
        self.last_checkpoint_time = time()
        os.makedirs(self.save_model_path, exist_ok=True)

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

    def get_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None:
        """(Optional) Get the learning rate scheduler."""
        return None

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

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler | None,
        batch: TensorTree,
    ) -> dict[str, float | str]:
        """Train step."""
        # TODO: think about ownership of .train and .zero_grad for safe override
        assert model.training, "Model must be in training mode"
        optimizer.zero_grad()
        loss, _ = self.get_loss(model, batch)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.grad_norm_max)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        return {"loss": loss.item(), "lr": current_lr, "grad_norm": grad_norm.item()}

    def val_step(self, model: nn.Module, batch: TensorTree) -> dict[str, float | str]:
        """Returns eval metrics."""
        assert not model.training, "Model must be in evaluation mode"
        loss, preds = self.get_loss(model, batch)
        metrics: dict[str, float | str] = {"loss": loss.item()}
        metrics.update(self.val_metrics(model, batch, preds))
        return metrics

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        step: dict[Literal["train", "val"], int],
        metrics: dict[str, float | str],
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        """Save a checkpoint of the model and training state."""
        if self.config.overwrite_existing_checkpoints:
            for file in os.listdir(self.save_model_path):
                if file.endswith(".pt"):
                    os.remove(os.path.join(self.save_model_path, file))

        checkpoint_path = os.path.join(self.save_model_path, f"ckpt_epoch_{epoch}.pt")
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            metrics=metrics,
            scheduler=scheduler,
        )
        self.logger.log_text(
            "Checkpointing",
            f"Saved checkpoint at epoch {epoch}, step {step}",
            save_to_file=False,
            write_to_console=True,
        )

    def run(self) -> None:
        """Creates or loads training variables and begins training."""
        model = self.get_model().to(self.config.device)
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        train_loader = self.get_train_dataloader()
        val_loader = self.get_val_dataloader()

        # load checkpoint if specified
        start_epoch = 0
        if self.config.load_model_path is not None:
            epoch, step, metrics = load_checkpoint(
                self.config.load_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=self.config.device,
            )
            start_epoch = epoch
            self.logger.step = step
            if metrics is not None:
                self.logger.log_text(
                    "Checkpointing",
                    f"Loaded checkpoint at epoch {epoch}, step {step}",
                    save_to_file=False,
                    write_to_console=True,
                )

        epoch_dec = self.config.num_epochs - start_epoch
        training_samples = 0
        eval_steps = 0
        while epoch_dec != 0:
            epoch = self.config.num_epochs - epoch_dec
            for batch in train_loader:
                # run full evaluation every n samples
                if training_samples >= eval_steps * self.config.eval_every_n_samples:
                    model.eval()
                    with torch.no_grad():
                        # TODO: fully support Loggable type (including dicts)
                        eval_metrics: dict[str, float | str] = {}
                        eval_size = 0

                        for val_batch in val_loader:
                            val_batch = move_to_device(val_batch, self.config.device)
                            metrics = self.val_step(model, val_batch)

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
                    self.logger.log_metrics(
                        eval_metrics,
                        mode="val",
                        header={"epoch": epoch, "samples": training_samples},
                    )
                    eval_steps += 1

                # classic train step
                model.train()
                training_samples += self.config.batch_size
                batch = move_to_device(batch, self.config.device)
                metrics = self.train_step(model, optimizer, scheduler, batch)
                self.logger.log_metrics(
                    metrics,
                    mode="train",
                    header={"epoch": epoch, "samples": training_samples},
                )

            # save checkpoint every n seconds
            if (
                self.config.save_every_n_seconds > 0
                and time() - self.last_checkpoint_time >= self.config.save_every_n_seconds
            ):
                self.save_checkpoint(model, optimizer, epoch + 1, self.logger.step, metrics, scheduler)
                self.last_checkpoint_time = time()

            epoch_dec -= 1
