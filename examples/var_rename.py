"""Variable renaming datagen and training w/ pre-LN transformer."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from iluvattnshun.nn import MultilayerTransformer
from iluvattnshun.prompter import PromptConfig, Prompter
from iluvattnshun.trainer import Trainer, TrainerConfig
from iluvattnshun.types import TensorTree


@dataclass
class VariableRenamingConfig(PromptConfig, TrainerConfig):
    """Configuration for variable renaming prompts."""

    # model
    num_layers: int
    """Number of transformer layers."""
    d_model: int
    """Dimension of the model."""
    n_heads: int
    """Number of attention heads."""

    # data generation
    num_chains: int
    """Number of independent renaming chains."""
    chain_length: int
    """Maximum length of a renaming chain."""
    train_size: int
    """Number of training examples."""
    test_size: int
    """Number of test examples."""
    dataset_path: str
    """Path to the dataset."""


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
            unfilled_chains = [i for i in range(self.config.num_chains) if len(chains[i]) <= self.config.chain_length]
            if len(unfilled_chains) == 0:
                break

            sampled_chain = np.random.choice(unfilled_chains)
            most_recent_vars = [chain[-1] for chain in chains]
            old_var = chains[sampled_chain][-1]
            new_var = np.random.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in most_recent_vars])
            chains[sampled_chain].append(new_var)

            prompt += f"{old_var}>{new_var};"

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
            elif c == ">":
                tokens.append(36)
            elif c == "?":
                tokens.append(37)
            elif c == ";":
                tokens.append(38)
            else:
                raise ValueError(f"Unexpected character: {c}")
        return tokens


class VariableRenamingTrainer(Trainer[VariableRenamingConfig]):
    """Training decoder-only transformer for variable renaming."""

    def init_state(self) -> None:
        """Adding datasets and tokenizers to the trainer state."""
        self.prompter = VariableRenamingPrompter(self.config)
        self.ds_dict = self.prompter.make_dataset(
            self.config.dataset_path,
            train_size=self.config.train_size,
            test_size=self.config.test_size,
            seed=42,
        )
        self.ds_dict.set_format(type="torch", columns=["prompt_tokens", "answer_tokens", "prompt", "answer"])

    def get_model(self) -> nn.Module:
        """Get the model."""
        max_seq_len = self.config.chain_length * self.config.num_chains * 4 + 2
        model = MultilayerTransformer(
            vocab_size=39,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.num_layers,
            max_seq_len=max_seq_len,
        )
        return model

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Returns a basic Adam optimizer."""
        return optim.Adam(model.parameters(), lr=1e-3)

    def get_loss(self, model: nn.Module, batch: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the cross-entropy loss over the final token logits."""
        logits, _, _ = model(batch["prompt_tokens"])
        logits = logits[:, -1, :]  # (batch_size, vocab_size)
        answer = batch["answer_tokens"].squeeze()  # (batch_size,)
        return nn.functional.cross_entropy(logits, answer), logits

    def val_metrics(self, model: nn.Module, batch: TensorTree, preds: torch.Tensor) -> dict[str, float | str]:
        """Get additional validation metrics for a batch."""
        predicted_answers = preds.argmax(dim=-1)
        sample_prompt = batch["prompt"][0]
        sample_answer = batch["answer"][0]
        sample_predicted_answer = predicted_answers[0]
        sample_probability = torch.softmax(preds[0], dim=-1)[sample_predicted_answer]

        target = batch["answer_tokens"].to(predicted_answers.device)
        correct = (predicted_answers.unsqueeze(-1) == target).float()
        total_accuracy = torch.mean(correct).item()

        return {
            "sample_prompt": sample_prompt,
            "sample_answer": sample_answer,
            "predicted_answer": str(sample_predicted_answer.item()),
            "probability": sample_probability.item(),
            "accuracy": total_accuracy,
        }

    def get_train_dataloader(self) -> Iterable[TensorTree]:
        """Get the train dataloader."""
        return torch.utils.data.DataLoader(
            self.ds_dict["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            prefetch_factor=4,
            num_workers=4,
        )

    def get_val_dataloader(self) -> Iterable[TensorTree]:
        """Get the val dataloader."""
        return torch.utils.data.DataLoader(
            self.ds_dict["test"],
            batch_size=self.config.batch_size,
            shuffle=True,
            prefetch_factor=4,
            num_workers=4,
        )


if __name__ == "__main__":
    config = VariableRenamingConfig(
        num_layers=3,
        d_model=64 * 3,
        n_heads=3,
        train_size=100000,
        test_size=1000,
        num_chains=2,
        chain_length=32,
        num_epochs=1000,
        batch_size=512,
        eval_every_n_samples=1000000,
        log_every_n_seconds=3,
        dataset_path="data/var_rename",
        tensorboard_logdir="logs/var_rename",
    )
    trainer = VariableRenamingTrainer(config)
    trainer.run()
