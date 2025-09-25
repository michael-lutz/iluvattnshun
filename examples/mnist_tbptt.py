"""MNIST training with diffusion model."""

import math
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import parse_shape, rearrange
from sklearn.datasets import fetch_openml  # type: ignore

from iluvattnshun.logger import Loggable
from iluvattnshun.trainer import SupervisedTrainer, TrainerConfig
from iluvattnshun.types import TensorTree


def unsqueeze_as(tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """Add singleton dimensions to tensor to match target tensor's dimensionality."""
    assert tensor.ndim <= target_tensor.ndim
    while tensor.ndim < target_tensor.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int = 10000) -> None:
        super().__init__()
        self.register_buffer("embedding", self.make_embedding(dim, max_length))

    # type hints for registered buffers
    embedding: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input tensor.

        Args:
            x: Input tensor of shape (bsz,) with discrete indices

        Returns:
            Positional embeddings of shape (bsz, dim)
        """
        return self.embedding[x.long()]

    @staticmethod
    def make_embedding(dim: int, max_length: int = 10000) -> torch.Tensor:
        """Create sinusoidal positional embedding matrix."""
        embedding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding


class FFN(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.init_embed = nn.Linear(in_dim, embed_dim)
        self.time_embed = PositionalEmbedding(embed_dim)
        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, in_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FFN.

        Args:
            x: Input tensor
            t: Time tensor for conditioning

        Returns:
            Processed tensor
        """
        x = self.init_embed(x)
        t = self.time_embed(t)
        return self.model(x + t)


class BasicBlock(nn.Module):
    """
    BasicBlock: two 3x3 convs followed by a residual connection then ReLU.
    [He et al. CVPR 2016]

        BasicBlock(x) = ReLU( x + Conv3x3( ReLU( Conv3x3(x) ) ) )

    This version supports an additive shift parameterized by time and label.
    """

    def __init__(self, in_c: int, out_c: int, time_c: int, label_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_c, time_c),
            nn.ReLU(),
            nn.Linear(time_c, out_c),
        )
        self.mlp_label = nn.Sequential(
            nn.Linear(label_c, label_c),
            nn.ReLU(),
            nn.Linear(label_c, out_c),
        )
        if in_c == out_c:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_c)
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through the basic block.

        Args:
            x: Input feature map
            t: Time conditioning tensor
            y: Label conditioning tensor

        Returns:
            Output feature map
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out + unsqueeze_as(self.mlp_time(t), x) + unsqueeze_as(self.mlp_label(y), x))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + self.shortcut(x))
        return out


class SelfAttention2d(nn.Module):
    """
    Only implements the MultiHeadAttention component, not the PositionwiseFFN component.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.o_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2D self-attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor with residual connection
        """
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        q = rearrange(q, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        k = rearrange(k, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        v = rearrange(v, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        a = torch.einsum("b c s, b c t -> b s t", q, k) / self.dim**0.5
        a = self.dropout(torch.softmax(a, dim=-1))
        o = torch.einsum("b s t, b c t -> b c s", a, v)
        o = rearrange(o, "(b g) c (h w) -> b (g c) h w", g=self.num_heads, w=x.shape[-1])
        return x + self.o_conv(o)


class UNet(nn.Module):
    """
    Simple implementation that closely mimics the one by Phil Wang (lucidrains).
    """

    def __init__(self, in_dim: int, embed_dim: int, dim_scales: Tuple[int, ...], num_classes: int = 10) -> None:
        super().__init__()

        self.init_embed = nn.Conv2d(in_dim, embed_dim, 1)
        self.time_embed = PositionalEmbedding(embed_dim)
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Example:
        #   in_dim=1, embed_dim=32, dim_scales=(1, 2, 4, 8) => all_dims=(32, 32, 64, 128, 256)
        all_dims = (embed_dim, *[embed_dim * s for s in dim_scales])

        for idx, (in_c, out_c) in enumerate(
            zip(
                all_dims[:-1],
                all_dims[1:],
            )
        ):
            is_last = idx == len(all_dims) - 2
            self.down_blocks.extend(
                nn.ModuleList(
                    [
                        BasicBlock(in_c, in_c, embed_dim, embed_dim),
                        BasicBlock(in_c, in_c, embed_dim, embed_dim),
                        nn.Conv2d(in_c, out_c, 3, 2, 1) if not is_last else nn.Conv2d(in_c, out_c, 1),
                    ]
                )
            )

        for idx, (in_c, out_c, skip_c) in enumerate(
            zip(
                all_dims[::-1][:-1],
                all_dims[::-1][1:],
                all_dims[:-1][::-1],
            )
        ):
            is_last = idx == len(all_dims) - 2
            self.up_blocks.extend(
                nn.ModuleList(
                    [
                        BasicBlock(in_c + skip_c, in_c, embed_dim, embed_dim),
                        BasicBlock(in_c + skip_c, in_c, embed_dim, embed_dim),
                        nn.ConvTranspose2d(in_c, out_c, 2, 2) if not is_last else nn.Conv2d(in_c, out_c, 1),
                    ]
                )
            )

        self.mid_blocks = nn.ModuleList(
            [
                BasicBlock(all_dims[-1], all_dims[-1], embed_dim, embed_dim),
                SelfAttention2d(all_dims[-1]),
                BasicBlock(all_dims[-1], all_dims[-1], embed_dim, embed_dim),
            ]
        )
        self.out_blocks = nn.ModuleList(
            [
                BasicBlock(embed_dim, embed_dim, embed_dim, embed_dim),
                nn.Conv2d(embed_dim, in_dim, 1, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            t: Time conditioning tensor
            y: Label conditioning tensor

        Returns:
            Output tensor of same shape as input
        """
        x = self.init_embed(x)
        t = self.time_embed(t)
        y = self.label_embed(y)
        skip_conns = []
        residual = x.clone()

        for block in self.down_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t, y)
                skip_conns.append(x)
            else:
                x = block(x)
        for block in self.mid_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t, y)
            else:
                x = block(x)
        for block in self.up_blocks:
            if isinstance(block, BasicBlock):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = block(x, t, y)
            else:
                x = block(x)

        x = x + residual
        for block in self.out_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t, y)
            else:
                x = block(x)
        return x


@dataclass
class MNISTConfig(TrainerConfig):
    """Configuration for MNIST DDIM training."""

    # Model architecture parameters
    model_channels: int = 128
    """Number of channels in the model."""
    channel_mult: tuple[int, ...] = (1, 2, 4, 8)
    """Channel multipliers for different resolution levels."""

    # Diffusion parameters
    train_timesteps: int = 500
    """Number of timesteps for sampling during training."""
    sample_timesteps: int = 500
    """Number of timesteps for sampling during evaluation."""

    # Training parameters
    learning_rate: float = 1e-3
    """Learning rate for optimizer. (1e-3 per Tony Duan's defaults)"""
    weight_decay: float = 0.0
    """Weight decay for optimizer. (0.0 per Tony Duan's defaults)"""

    # Data parameters
    image_size: int = 32
    """Size of the images (will be padded from 28x28 to 32x32)."""


class DDIMModel(nn.Module):
    """DDIM wrapper that includes loss and sampling for underlying model.

    Note that we make the following simplifying assumptions:
    - "Simple" L2 loss
    - Cosine noise schedule
    - Deterministic sampling
    - Image-like inputs shaped (batch, channels, height, width)
    """

    # type hints for registered buffers
    alpha_t: torch.Tensor
    sigma_t: torch.Tensor

    def __init__(
        self,
        nn_module: nn.Module,
        train_timesteps: int,
        input_shape: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.nn_module = nn_module
        self.train_timesteps = train_timesteps
        self.input_shape = input_shape

        # cosine noise schedule and registering as in-mem buffer
        linspace = torch.linspace(0, 1, self.train_timesteps + 1)
        f_t = torch.cos((linspace + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
        bar_alpha_t = f_t / f_t[0]
        beta_t = torch.zeros_like(bar_alpha_t)
        beta_t[1:] = (1 - (bar_alpha_t[1:] / bar_alpha_t[:-1])).clamp(min=0, max=0.999)
        alpha_t = torch.cumprod(1 - beta_t, dim=0) ** 0.5

        # reshaping to (timesteps + 1, 1, 1, 1) for channel, height, width
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = (1 - alpha_t**2).clamp(min=0) ** 0.5

        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("sigma_t", sigma_t)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            y: Label tensor of shape (batch,)

        Returns:
            Scalar loss tensor
        """
        batch_size, channels, height, width = x.shape
        device = x.device
        # t_sample = torch.randint(1, self.train_timesteps + 1, size=(B,), device=x.device)
        # eps = torch.randn_like(x)
        # x_t = self.alpha_t[t_sample] * x + self.sigma_t[t_sample] * eps
        # pred_target = self.nn_module(x_t, t_sample)

        gt_target = x

        x = torch.randn((batch_size, *self.input_shape), device=device)
        t_start = torch.empty((batch_size,), dtype=torch.int64, device=device)
        t_end = torch.empty((batch_size,), dtype=torch.int64, device=device)
        subseq = torch.linspace(self.train_timesteps, 0, self.train_timesteps + 1).round()

        # Note that t_start > t_end we're traversing pairwise down subseq.
        # For example, subseq here could be [500, 400, 300, 200, 100, 0]
        losses = []
        for idx, (scalar_t_start, scalar_t_end) in enumerate(zip(subseq[:-1], subseq[1:])):

            t_start.fill_(scalar_t_start)
            t_end.fill_(scalar_t_end)

            nn_out = self.nn_module(x, t_start, y)

            overall_loss = (gt_target - nn_out) ** 2
            # min_loss = torch.inf
            # for _batch in overall_loss:
            #     batch_loss = _batch.mean()
            #     if batch_loss < min_loss:
            #         min_loss = batch_loss

            losses.append(overall_loss.mean())

            # pred_eps = (x - self.alpha_t[t_start] * nn_out) / self.sigma_t[t_start]
            # x = (self.alpha_t[t_end] * pred_x_0) + (self.sigma_t[t_end] ** 2).clamp(min=0) ** 0.5 * pred_eps
            x = torch.detach(nn_out)

        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(
        self, batch_size: int, device: str, sample_timesteps: int, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            sample_timesteps: Number of timesteps for sampling
            labels: Optional labels to condition on. If None, generates random labels.

        Returns:
            samples: (num_sampling_timesteps + 1, bsz, *self.input_shape)
                index 0 corresponds to x_0
                index t corresponds to x_t
                last index corresponds to random noise
        """
        assert (
            1 <= sample_timesteps <= self.train_timesteps
        ), f"{sample_timesteps=} must be between 1 and {self.train_timesteps=}"

        if labels is None:
            labels = torch.randint(0, 10, (batch_size,), device=device)
        else:
            assert labels.shape[0] == batch_size, f"Expected {batch_size} labels, got {labels.shape[0]}"

        x = torch.randn((batch_size, *self.input_shape), device=device)
        t_start = torch.empty((batch_size,), dtype=torch.int64, device=device)
        t_end = torch.empty((batch_size,), dtype=torch.int64, device=device)

        subseq = torch.linspace(self.train_timesteps, 0, sample_timesteps + 1).round()
        samples = torch.empty((sample_timesteps + 1, batch_size, *self.input_shape), device=device)
        samples[-1] = x

        # Note that t_start > t_end we're traversing pairwise down subseq.
        # For example, subseq here could be [500, 400, 300, 200, 100, 0]
        for idx, (scalar_t_start, scalar_t_end) in enumerate(zip(subseq[:-1], subseq[1:])):

            t_start.fill_(scalar_t_start)
            t_end.fill_(scalar_t_end)

            nn_out = self.nn_module(x, t_start, labels)
            pred_x_0 = nn_out
            # pred_eps = (x - self.alpha_t[t_start] * nn_out) / self.sigma_t[t_start]
            # x = (self.alpha_t[t_end] * pred_x_0) + (self.sigma_t[t_end] ** 2).clamp(min=0) ** 0.5 * pred_eps
            x = torch.detach(pred_x_0)
            samples[-1 - idx - 1] = pred_x_0

        return samples


def load_mnist_data() -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load MNIST dataset and preprocess it.

    Returns:
        Tuple of (images, labels) where:
        - images: Preprocessed MNIST data as numpy array with shape (n_samples, 1, 32, 32)
        - labels: Integer labels with shape (n_samples,)
    """
    # (7000, 784) dataset, no train-test split (validating via generations)
    x, labels = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)

    # reshape to 32x32 with singleton channel dim
    x = rearrange(x, "b (h w) -> b h w", h=28, w=28)
    x = np.pad(x, pad_width=((0, 0), (2, 2), (2, 2)))
    x = rearrange(x, "b h w -> b () h w")  # add singleton channel dim

    # standardize to [-1, 1]
    input_mean = np.full((1, 1, 32, 32), fill_value=127.5, dtype=np.float32)
    input_sd = np.full((1, 1, 32, 32), fill_value=127.5, dtype=np.float32)
    x = ((x - input_mean) / input_sd).astype(np.float32)

    # convert string labels to integers
    labels_int = labels.astype(int)

    return x, labels_int


class MNISTTrainer(SupervisedTrainer[MNISTConfig]):
    """Training diffusion model for MNIST generation."""

    def init_state(self) -> None:
        """Initialize datasets and preprocessing parameters."""
        self.train_data: np.ndarray[Any, Any]
        self.train_labels: np.ndarray[Any, Any]
        self.train_tensor: torch.Tensor
        self.train_labels_tensor: torch.Tensor

        self.train_data, self.train_labels = load_mnist_data()
        self.train_tensor = torch.from_numpy(self.train_data).to(self.config.device)
        self.train_labels_tensor = torch.from_numpy(self.train_labels).to(self.config.device)

    def get_model(self) -> DDIMModel:
        """Get the diffusion model."""
        model = UNet(
            in_dim=1,
            embed_dim=self.config.model_channels,
            dim_scales=self.config.channel_mult,
            num_classes=10,
        )

        return DDIMModel(
            nn_module=model,
            train_timesteps=self.config.train_timesteps,
            input_shape=(1, self.config.image_size, self.config.image_size),
        )

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Returns Adam optimizer."""
        return optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def get_lr_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None:
        """Returns cosine annealing scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.num_epochs)

    def get_loss(self, model: nn.Module, batch: TensorTree) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the diffusion loss."""
        images = batch["images"]
        labels = batch["labels"]
        # model is a DDIMModel which has a loss method
        loss = model.loss(images, labels)  # type: ignore
        return loss, images

    def val_step(self, model: nn.Module, batch: TensorTree) -> dict[str, Loggable]:
        """Validation step with basic metrics."""
        return {}

    def post_val_step(self, model: nn.Module) -> dict[str, Loggable]:
        """Post-validation step with image generation."""
        with torch.no_grad():
            # generate samples for visualization with specific labels (0-9)
            labels = torch.arange(10, device=self.config.device)
            samples = model.sample(  # type: ignore
                batch_size=10, device=self.config.device, sample_timesteps=self.config.sample_timesteps, labels=labels
            )

            output_tensor = None
            for i in range(self.config.sample_timesteps + 1):
                # get final generated samples (x_0) and arrange in a grid
                final_samples = samples[i]  # first index is x_0

                # create a grid of generated images (1x10)
                grid_samples = final_samples[:10]  # take first 10 samples
                # reshape to (1, 10*32, 32) for horizontal layout
                grid = grid_samples.view(10, 1, 32, 32)  # (10, 1, 32, 32)
                grid = grid.permute(0, 2, 1, 3)  # (10, 32, 1, 32)
                grid = grid.contiguous().view(10 * 32, 32)  # (10*32, 32)
                grid = grid.unsqueeze(0)  # add batch dimension: (1, 10*32, 32)

                # convert to [0, 1] range for proper display
                grid = (grid + 1) / 2  # convert from [-1, 1] to [0, 1]
                grid = grid.clamp(0, 1)

                output_tensor = torch.cat((output_tensor, grid), dim=2) if output_tensor is not None else grid

            return {"generated_samples": output_tensor}

    def get_train_dataloader(self) -> Iterable[TensorTree]:
        """Training with random sampling from training data."""
        while True:
            indices = torch.randint(0, len(self.train_tensor), (self.config.batch_size,))
            batch_images = self.train_tensor[indices]
            batch_labels = self.train_labels_tensor[indices]
            yield {"images": batch_images, "labels": batch_labels}

    def get_val_dataloader(self) -> Iterable[TensorTree]:
        """Not validating by traditional MSE loss."""
        return []


if __name__ == "__main__":
    config = MNISTConfig(
        model_channels=64,
        channel_mult=(1, 2, 4, 8),
        train_timesteps=10,
        sample_timesteps=10,
        learning_rate=1e-3,
        weight_decay=0.0,
        image_size=32,
        num_epochs=100,
        batch_size=64,
        eval_every_n_samples=1000,
        log_every_n_seconds=3,
        tensorboard_logdir="logs/mnist",
    )
    trainer = MNISTTrainer(config)
    trainer.run()
