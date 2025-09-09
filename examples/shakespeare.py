"""Shakespeare training w/ pre-LN transformer."""

import math
from dataclasses import dataclass
from typing import Iterable

import tensorflow_datasets as tfds  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim

from iluvattnshun.nn import TokenTransformer, TransformerLayer
from iluvattnshun.trainer import SupervisedTrainer, TrainerConfig
from iluvattnshun.types import TensorTree


@dataclass
class ShakespeareConfig(TrainerConfig):
    """Configuration for shakespeare training."""

    num_layers: int
    """Number of transformer layers."""
    max_context_length: int
    """Maximum context length for the transformer."""
    d_model: int
    """Dimension of the model."""
    n_heads: int
    """Number of attention heads."""
    rope_base: int
    """Base for the rotary positional embedding."""


def load_shakespeare_text(split: str = "train") -> str:
    """Loads the Tiny Shakespeare dataset.

    This function loads the tiny_shakespeare dataset from tfds, extracts the
    text, and builds a character-level tokenizer.

    Returns:
        The loaded dataset.
    """
    ds = tfds.load("tiny_shakespeare", split=split, as_supervised=False)

    text = ""
    for example in tfds.as_numpy(ds):
        text += example["text"].decode("utf-8")

    return text


class HeirarchicalReasoner(nn.Module):
    """A heirarchical reasoner (https://arxiv.org/abs/2506.21734)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        rope_base: float = 10000.0,
        dropout_attn: float = 0.1,
        dropout_mlp: float = 0.1,
        dropout_emb: float = 0.1,
        l_steps: int = 2,
        h_steps: int = 2,
    ):
        super().__init__()

        self.l_steps = l_steps
        self.h_steps = h_steps

        # For now, keep low level nn the same as the high level nn
        self.low_level_nn = nn.Sequential(
            *[TransformerLayer(d_model, n_heads, rope_base, dropout_attn=0.1, dropout_mlp=0.1) for _ in range(n_layers)]
        )
        self.high_level_nn = nn.Sequential(
            *[TransformerLayer(d_model, n_heads, rope_base, dropout_attn=0.1, dropout_mlp=0.1) for _ in range(n_layers)]
        )

    def forward(self, x_l: torch.Tensor, x_h: torch.Tensor, x_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through hierarchical latent reasoner.

        With no gradient: L -> ... -> L -> H -> L -> ... -> L -> H -> ...
        With gradient:    L -> H
        """
        # TODO: experiment with reversing the order (maintain 1-step grad)

        # let the gradients flow through the input
        with torch.no_grad():
            for i_h in range(self.h_steps):
                for i_l in range(self.l_steps):
                    # skip grad until the last low level call
                    if not (i_h == self.h_steps - 1 and i_l == self.l_steps - 1):
                        x_l = self.low_level_nn((x_l + x_h + x_input) / 3)  # div by 3 to keep init variance

                        # TODO: try += and add LayerNorm

                # skip grad until the last high level call
                if i_h != self.h_steps - 1:
                    x_h = self.high_level_nn((x_l + x_h) / 2)

        # 1-step grad approximation
        x_l = self.low_level_nn((x_l + x_h + x_input) / 3)
        x_h = self.high_level_nn((x_l + x_h) / 2)

        return x_l, x_h


class HeirarchicalLanguageModel(nn.Module):
    """A heirarchical language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        rope_base: float = 10000.0,
        dropout_attn: float = 0.1,
        dropout_mlp: float = 0.1,
        dropout_emb: float = 0.1,
        l_steps: int = 2,
        h_steps: int = 2,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, scale_grad_by_freq=True)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model**-0.5)
        self.dropout_emb = nn.Dropout(dropout_emb)
        self.l_init = nn.Parameter(torch.randn(1, 1, d_model) * d_model**-0.5)
        self.h_init = nn.Parameter(torch.randn(1, 1, d_model) * d_model**-0.5)
        self.reasoner = HeirarchicalReasoner(
            vocab_size, d_model, n_heads, n_layers, rope_base, dropout_attn, dropout_mlp, dropout_emb, l_steps, h_steps
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hierarchical language model."""
        x_input = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        x_input = self.dropout_emb(x_input)

        x_l = self.l_init.repeat(x.shape[0], 1, 1)
        x_h = self.h_init.repeat(x.shape[0], 1, 1)
        x_l, x_h = self.reasoner(x_l, x_h, x_input)

        return self.output(x_h)


class ShakespeareTrainer(SupervisedTrainer[ShakespeareConfig]):
    """Training decoder-only transformer for Shakespeare text."""

    def init_state(self) -> None:
        """Adding datasets and tokenizers to the trainer state."""
        self.train_ds = load_shakespeare_text(split="train")
        self.val_ds = load_shakespeare_text(split="validation")
        unique_tokens = sorted(list(set(self.train_ds + self.val_ds)))
        self.token_to_id = {ch: i for i, ch in enumerate(unique_tokens)}
        self.id_to_token = {i: ch for i, ch in enumerate(unique_tokens)}

        # tiny_shakespeare is small, so we store in GPU mem for faster access
        self.train_token_ids = torch.tensor([self.token_to_id[c] for c in self.train_ds])
        self.val_token_ids = torch.tensor([self.token_to_id[c] for c in self.val_ds])
        self.train_token_ids.to(self.config.device)
        self.val_token_ids.to(self.config.device)

    def get_model(self) -> nn.Module:
        """Get the model."""
        return TokenTransformer(
            vocab_size=len(self.token_to_id),
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.num_layers,
            rope_base=self.config.rope_base,
        )

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Returns a basic Adam optimizer."""
        return optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)

    def get_loss(self, model: nn.Module, batch: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the cross-entropy loss over the final token logits."""
        logits, _, _, _ = model(batch["prompt_tokens"])  # (batch_size, seq_len, vocab_size)
        target = batch["answer_tokens"]  # (batch_size, seq_len)

        # flatten for cross entropy
        _, _, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        return nn.functional.cross_entropy(logits_flat, target_flat), logits

    def val_metrics(self, model: nn.Module, batch: TensorTree, preds: torch.Tensor) -> dict[str, float | str]:
        """Get additional validation metrics for a batch."""
        predicted_chars = preds.argmax(dim=-1)
        target = batch["answer_tokens"].to(predicted_chars.device)
        total_accuracy = torch.mean((predicted_chars == target).float()).item()

        return {
            "accuracy": total_accuracy,
        }

    def post_val_metrics(self, model: nn.Module) -> dict[str, float | str]:
        """Get additional validation metrics for a batch."""
        assert isinstance(model, TokenTransformer), "Making mypy happy"
        prompt = "tomorrow"  # and tomorrow and tomorrow...
        prompt_tokens = torch.tensor([self.token_to_id[c] for c in prompt]).to(self.config.device)
        prompt_tokens = prompt_tokens.unsqueeze(0)
        generated = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8)
        generated_text = "".join(self.id_to_token[int(t.item())] for t in generated[0])
        return {
            "generated_text": generated_text,
        }

    def get_train_dataloader(self) -> Iterable[TensorTree]:
        """Training with IID sampling (indefinitely)."""
        while True:
            idx = torch.randint(
                0,
                self.train_token_ids.shape[0] - self.config.max_context_length - 1,
                (self.config.batch_size,),
            )
            windows = idx.unsqueeze(1) + torch.arange(self.config.max_context_length)
            yield {
                "prompt_tokens": self.train_token_ids[windows],
                "answer_tokens": self.train_token_ids[windows + 1],
            }

    def get_val_dataloader(self) -> Iterable[TensorTree]:
        """Performing validation with deterministic batches."""
        num_batches = math.ceil(self.val_token_ids.shape[0] / self.config.batch_size)
        batched_val_ds = []
        for i in range(num_batches):
            idx = torch.arange(i * self.config.batch_size, (i + 1) * self.config.batch_size)
            windows = idx.unsqueeze(1) + torch.arange(self.config.max_context_length)

            # naive truncation
            windows[windows >= self.val_token_ids.shape[0] - 1] = 0

            batched_val_ds.append(
                {
                    "prompt_tokens": self.val_token_ids[windows],
                    "answer_tokens": self.val_token_ids[windows + 1],
                }
            )

        return batched_val_ds


if __name__ == "__main__":
    config = ShakespeareConfig(
        num_layers=8,
        d_model=128,
        n_heads=4,
        rope_base=1024,
        max_context_length=128,
        num_epochs=-1,  # using IID sampling, not epochal training
        batch_size=64,
        eval_every_n_samples=20000,
        log_every_n_seconds=3,
        tensorboard_logdir="logs/shakespeare",
    )
    trainer = ShakespeareTrainer(config)
    trainer.run()
