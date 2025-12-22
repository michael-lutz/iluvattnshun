"""Training GPT-OSS-20B on Shakespeare dataset using iluvattnshun trainer."""

import math
from dataclasses import dataclass
from typing import Iterable

import tensorflow_datasets as tfds  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim

from iluvattnshun.logger import Loggable
from iluvattnshun.trainer import SupervisedTrainer, TrainerConfig
from iluvattnshun.types import TensorTree

from model.model import Transformer, ModelConfig, TransformerBlock
from model.tokenizer import get_tokenizer


@dataclass
class GPTShakespeareConfig(TrainerConfig):
    """Configuration for GPT-OSS-20B Shakespeare training."""

    checkpoint_path: str = "../../weights/gpt-oss-20b/original"
    """Path to pretrained model checkpoint."""

    max_context_length: int = 256
    """Maximum context length for training sequences."""

    learning_rate: float = 1e-5
    """Learning rate for optimizer."""

    weight_decay: float = 1e-2
    """Weight decay for optimizer."""

    freeze_base_model: bool = False
    """If True, freeze all base model layers and only train additional layer."""

    add_extra_layer: bool = False
    """If True, add an extra transformer layer at the end."""

    temperature: float = 0.8
    """Temperature for text generation."""

    max_gen_tokens: int = 100
    """Maximum tokens to generate during validation."""


def load_shakespeare_text(split: str = "train") -> str:
    """Loads the Tiny Shakespeare dataset.

    Returns:
        The loaded dataset text.
    """
    ds = tfds.load("tiny_shakespeare", split=split, as_supervised=False)

    text = ""
    for example in tfds.as_numpy(ds):
        text += example["text"].decode("utf-8")

    return text


class GPTOSSWithExtraLayer(nn.Module):
    """GPT-OSS model wrapper that can add an extra trainable layer."""

    def __init__(self, base_model: Transformer, add_extra_layer: bool = False):
        super().__init__()
        self.base_model = base_model
        self.add_extra_layer = add_extra_layer

        if add_extra_layer:
            # Get config from base model
            config = ModelConfig()
            self.extra_layer = TransformerBlock(
                config,
                layer_idx=config.num_hidden_layers,  # New layer index
                device=next(base_model.parameters()).device
            )
        else:
            self.extra_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        # Run through embedding
        x = self.base_model.embedding(x)

        # Run through base transformer blocks
        for block in self.base_model.block:
            x = block(x)

        # Optionally run through extra layer
        if self.extra_layer is not None:
            x = self.extra_layer(x)

        # Final norm and unembedding
        x = self.base_model.norm(x)
        x = self.base_model.unembedding(x)
        return x


class GPTShakespeareTrainer(SupervisedTrainer[GPTShakespeareConfig]):
    """Training GPT-OSS-20B for Shakespeare text generation."""

    def init_state(self) -> None:
        """Initialize datasets and tokenizer."""
        # Load datasets
        self.train_ds = load_shakespeare_text(split="train")
        self.val_ds = load_shakespeare_text(split="validation")

        # Load tokenizer
        self.tokenizer = get_tokenizer()

        # Tokenize datasets - store as lists since they might be too large for single tensor
        print("Tokenizing training data...")
        self.train_token_ids = self.tokenizer.encode(self.train_ds)
        print(f"Training tokens: {len(self.train_token_ids)}")

        print("Tokenizing validation data...")
        self.val_token_ids = self.tokenizer.encode(self.val_ds)
        print(f"Validation tokens: {len(self.val_token_ids)}")

        # Convert to tensors and move to device for faster access
        self.train_token_ids = torch.tensor(self.train_token_ids, dtype=torch.long)
        self.val_token_ids = torch.tensor(self.val_token_ids, dtype=torch.long)

    def get_model(self) -> nn.Module:
        """Load pretrained model and optionally add extra layer."""
        print(f"Loading pretrained model from {self.config.checkpoint_path}...")
        base_model = Transformer.from_checkpoint(
            self.config.checkpoint_path,
            device=self.config.device
        )
        print("Model loaded!")

        # Wrap model with extra layer if needed
        model = GPTOSSWithExtraLayer(
            base_model,
            add_extra_layer=self.config.add_extra_layer
        )

        # Freeze base model parameters if requested
        if self.config.freeze_base_model:
            print("Freezing base model parameters...")
            for param in base_model.parameters():
                param.requires_grad = False

            # Only train the extra layer
            if self.config.add_extra_layer and model.extra_layer is not None:
                print("Only training extra layer parameters")
                for param in model.extra_layer.parameters():
                    param.requires_grad = True
            else:
                print("WARNING: freeze_base_model=True but add_extra_layer=False. No parameters to train!")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        return model

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Returns AdamW optimizer."""
        # Only optimize parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def get_loss(self, model: nn.Module, batch: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns cross-entropy loss over next-token prediction."""
        logits = model(batch["input_ids"])  # (batch_size, seq_len, vocab_size)
        target = batch["labels"]  # (batch_size, seq_len)

        # Flatten for cross entropy
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        loss = nn.functional.cross_entropy(logits_flat, target_flat)
        return loss, logits

    def val_step(self, model: nn.Module, batch: TensorTree) -> dict[str, Loggable]:
        """Validation step with loss and accuracy."""
        with torch.no_grad():
            loss, logits = self.get_loss(model, batch)

            # Compute accuracy
            predicted_tokens = logits.argmax(dim=-1)
            target = batch["labels"]
            accuracy = (predicted_tokens == target).float().mean().item()

            return {
                "loss": loss.item(),
                "accuracy": accuracy,
            }

    def post_val_step(self, model: nn.Module) -> dict[str, Loggable]:
        """Generate sample text after validation."""
        with torch.no_grad():
            model.eval()

            # Generate from a classic Shakespeare prompt
            prompt = "To be, or not to be"
            prompt_tokens = self.tokenizer.encode(prompt)

            # Generate tokens
            generated_tokens = list(prompt_tokens)
            input_ids = torch.tensor([generated_tokens], dtype=torch.long, device=self.config.device)

            for _ in range(self.config.max_gen_tokens):
                # Get logits for next token
                logits = model(input_ids)
                next_token_logits = logits[0, -1, :]

                # Sample with temperature
                if self.config.temperature > 0:
                    probs = torch.softmax(next_token_logits / self.config.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(next_token_logits).item()

                generated_tokens.append(next_token)

                # Update input_ids
                input_ids = torch.tensor([generated_tokens], dtype=torch.long, device=self.config.device)

                # Stop at end of text token
                if next_token == self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]:
                    break

            # Decode generated text
            generated_text = self.tokenizer.decode(generated_tokens)

            return {"generated_text": generated_text}

    def get_train_dataloader(self) -> Iterable[TensorTree]:
        """Training with random sampling (indefinitely)."""
        while True:
            # Sample random starting positions
            max_start = len(self.train_token_ids) - self.config.max_context_length - 1
            if max_start <= 0:
                raise ValueError("Dataset too small for given context length")

            idx = torch.randint(0, max_start, (self.config.batch_size,))

            # Create sequences
            windows = idx.unsqueeze(1) + torch.arange(self.config.max_context_length)

            input_ids = self.train_token_ids[windows].to(self.config.device)
            labels = self.train_token_ids[windows + 1].to(self.config.device)

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }

    def get_val_dataloader(self) -> Iterable[TensorTree]:
        """Validation with deterministic batches."""
        num_samples = len(self.val_token_ids) - self.config.max_context_length - 1
        num_batches = math.ceil(num_samples / self.config.batch_size)

        batched_val_ds = []
        for i in range(num_batches):
            start_idx = i * self.config.batch_size
            end_idx = min((i + 1) * self.config.batch_size, num_samples)
            batch_size = end_idx - start_idx

            idx = torch.arange(start_idx, end_idx)
            windows = idx.unsqueeze(1) + torch.arange(self.config.max_context_length)

            # Ensure we don't go out of bounds
            windows = windows.clamp(max=len(self.val_token_ids) - 2)

            input_ids = self.val_token_ids[windows].to(self.config.device)
            labels = self.val_token_ids[windows + 1].to(self.config.device)

            batched_val_ds.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        return batched_val_ds


if __name__ == "__main__":
    # Configuration for full finetune with tiny batch size
    config = GPTShakespeareConfig(
        checkpoint_path="../../weights/gpt-oss-20b/original",
        max_context_length=256,
        learning_rate=1e-5,
        weight_decay=1e-2,
        num_epochs=-1,  # Infinite training with IID sampling
        batch_size=1,  # Start with tiny batch size
        eval_every_n_samples=100,  # Evaluate frequently
        log_every_n_seconds=10,
        tensorboard_logdir="logs/gpt_shakespeare",
        save_every_n_seconds=600,  # Save every 10 minutes
        freeze_base_model=False,
        add_extra_layer=False,
        temperature=0.8,
        max_gen_tokens=100,
    )

    trainer = GPTShakespeareTrainer(config)
    trainer.run()
