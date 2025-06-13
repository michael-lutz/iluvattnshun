import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset

from iluvattnshun.prompter import PromptConfig, Prompter
from iluvattnshun.trainer import Trainer, TrainerConfig
from iluvattnshun.types import TensorTree


@dataclass
class VariableRenamingConfig(PromptConfig, TrainerConfig):
    """Configuration for variable renaming prompts.

    Parameters:
        num_chains: Number of independent renaming chains
        depth: Maximum variable renaming chain length
    """

    num_chains: int
    depth: int
    dataset_path: str


class VariableRenamingPrompter(Prompter[VariableRenamingConfig]):
    """Prompter for generating variable renaming exercises.

    Generates prompts where variables are renamed in chains, and the task
    is to evaluate the final variable in terms of the initial value.
    """

    def get_prompt(self) -> tuple[str, str]:
        """Samples a variable renaming prompt and answers.

        Key idea is that when redefining, only sample from variables which are
        not currently the most recent variable in any chain.

        A key limitation is that we don't allow for more than 25 chains.
        """
        assert self.config.num_chains <= 25, "We don't support more than 25 chains."

        chains: list[list[int | str]] = [[np.random.randint(0, 10)] for _ in range(self.config.num_chains)]
        prompt = ""

        while True:
            unfilled_chains = [i for i in range(self.config.num_chains) if len(chains[i]) < self.config.depth + 1]
            if len(unfilled_chains) == 0:
                break

            sampled_chain = np.random.choice(unfilled_chains)
            most_recent_vars = [chain[-1] for chain in chains]
            old_var = chains[sampled_chain][-1]
            new_var = np.random.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in most_recent_vars])
            chains[sampled_chain].append(new_var)

            prompt += f"{new_var}={old_var};"

        final_var_evals = [(str(chain[-1]), str(chain[0])) for chain in chains]
        var_to_eval = np.random.choice(len(final_var_evals))
        prompt += final_var_evals[var_to_eval][0] + "?"
        answer = final_var_evals[var_to_eval][1]

        return prompt, answer

    def tokenize(self, text: str) -> list[int]:
        """Tokenize the input text.

        Maps:
        - Numbers 0-9 -> tokens 0-9
        - Letters a-z -> tokens 10-35
        - '=' -> token 36
        - '?' -> token 37
        - ';' -> token 38
        """
        tokens = []
        for c in text:
            if c.isdigit():
                tokens.append(int(c))
            elif c.isalpha() and c.islower():
                tokens.append(ord(c) - ord("a") + 10)
            elif c == "=":
                tokens.append(36)
            elif c == "?":
                tokens.append(37)
            elif c == ";":
                tokens.append(38)
            else:
                raise ValueError(f"Unexpected character: {c}")
        return tokens


class TransformerLayer(nn.Module):
    """A single transformer layer."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer layer."""
        x_norm = self.norm1(x)  # Pre-LN Xiong et al., 2020 (https://arxiv.org/abs/2002.04745v1)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x


class MultilayerTransformer(nn.Module):
    """A multilayer transformer model."""

    def __init__(
        self, vocab_size: int = 39, d_model: int = 128, n_heads: int = 1, n_layers: int = 4, max_seq_len: int = 102
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, scale_grad_by_freq=True)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model**-0.5)  # Scale during initialization
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.pos_embedding(pos)

        for layer in self.layers:
            x = layer(x)

        logits = self.output(x)
        return logits


class VariableRenamingTrainer(Trainer[VariableRenamingConfig]):
    def get_model(self) -> nn.Module:
        return MultilayerTransformer()

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=1e-3)

    def get_loss(self, model: nn.Module, batch: TensorTree) -> torch.Tensor:
        logits = model(batch["prompt_tokens"])[:, -1, :]
        answer = batch["answer_tokens"]
        return nn.functional.cross_entropy(logits, answer)

    def get_train_dataloader(self) -> Iterable[TensorTree]:
        train_path = Path(self.config.dataset_path) / "train"
        if not os.path.exists(train_path):
            prompter = VariableRenamingPrompter(self.config)
            prompter.make_dataset(train_path.as_posix())

        dataset = Dataset.load_from_disk(train_path.as_posix())
        dataset.set_format(type="torch", columns=["prompt_tokens", "answer_tokens"])
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def get_val_dataloader(self) -> Iterable[TensorTree]:
        val_path = Path(self.config.dataset_path) / "val"
        if not os.path.exists(val_path):
            prompter = VariableRenamingPrompter(self.config)
            prompter.make_dataset(val_path.as_posix())

        dataset = Dataset.load_from_disk(val_path.as_posix())
        dataset.set_format(type="torch", columns=["prompt_tokens", "answer_tokens"])
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)


if __name__ == "__main__":
    config = VariableRenamingConfig(
        num_prompts=100,
        num_chains=5,
        depth=5,
        seed=42,
        num_epochs=10,
        batch_size=128,
        log_every_n_seconds=10,
        log_fp=4,
        dataset_path="data/var_rename",
    )
    trainer = VariableRenamingTrainer(config)
    trainer.run()
