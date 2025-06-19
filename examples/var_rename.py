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

MASK_TOKEN = "."
MASK_ID = 38


@dataclass
class VariableRenamingConfig(PromptConfig, TrainerConfig):
    """Configuration for variable renaming prompts."""

    # model
    num_layers: int
    """Number of transformer layers."""
    dim_model: int
    """Dimension of the model."""
    num_heads: int
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

    def data_hash_params(self) -> list[str]:
        return ["num_chains", "chain_length", "train_size", "test_size"]


class VariableRenamingPrompter(Prompter[VariableRenamingConfig]):
    """Prompter for generating variable renaming exercises.

    Generates prompts where variables are renamed in chains, and the task
    is to evaluate the final variable in terms of the initial value.
    """

    def get_prompt(self) -> tuple[str, str]:
        """Samples a variable renaming prompt and answers.

        Example prompt: "|1>a|2>b|a>c|b>d"
        Answers:        ".1.1.2.2.1.1.2.2" where "." means <MASK>

        When redefining, only sample from variables which are not currently at
        the end of any chain (no DAG structure).

        Since evaluations are single digit, we can only have at most 10 chains.
        """
        assert self.config.num_chains <= 10, "We don't support more than 10 chains."

        # TODO: ensure no resampling of same number
        evaluations = list(range(self.config.num_chains))
        np.random.shuffle(evaluations)

        chains: list[list[int | str]] = [[evaluations[i]] for i in range(self.config.num_chains)]
        prompt = ""
        answers = ""

        while True:
            unfilled_chains = [i for i in range(self.config.num_chains) if len(chains[i]) <= self.config.chain_length]
            if len(unfilled_chains) == 0:
                break

            sampled_chain = np.random.choice(unfilled_chains)
            most_recent_vars = [chain[-1] for chain in chains]
            old_var = chains[sampled_chain][-1]
            new_var = np.random.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in most_recent_vars])
            chains[sampled_chain].append(new_var)
            evaluation = chains[sampled_chain][0]

            prompt += f"|{old_var}>{new_var}"
            answers += f".{evaluation}.{evaluation}"

        return prompt, answers

    @property
    def _tokenization_map(self) -> dict[str, int]:
        char_to_token: dict[str, int] = {}
        for i in range(10):
            char_to_token[str(i)] = i
        for c in "abcdefghijklmnopqrstuvwxyz":
            char_to_token[c] = ord(c) - ord("a") + 10
        char_to_token[">"] = 36
        char_to_token["|"] = 37
        char_to_token["."] = 38
        return char_to_token

    def tokenize(self, text: str) -> list[int]:
        """Tokenize the input text.

        Maps:
        - Numbers 0-9 -> tokens 0-9
        - Letters a-z -> tokens 10-35
        - '>' -> token 36
        - '|' -> token 37
        - '.' -> token 38 (mask)
        """
        tokens = []
        for c in text:
            tokens.append(self._tokenization_map[c])
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        """Detokenize the input tokens.

        Maps:
        - Tokens 0-9 -> Numbers 0-9
        - Tokens 10-35 -> Letters a-z
        - Tokens 36-37 -> '>', '|'
        - Token 38 -> '.' (mask)
        """
        inverse_tokenization_map: dict[int, str] = {v: k for k, v in self._tokenization_map.items()}
        return "".join([inverse_tokenization_map[token] for token in tokens])


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

        # log train test stats
        train_prompts = set(self.ds_dict["train"]["prompt"])
        test_prompts = set(self.ds_dict["test"]["prompt"])
        train_total = len(self.ds_dict["train"])
        train_unique = len(train_prompts)
        train_duplicates = train_total - train_unique
        overlap_with_train = test_prompts.intersection(train_prompts)
        self.logger.log_text(
            "train_test_stats.txt",
            f"[Train Set] Total: {train_total}, Unique: {train_unique}, Duplicates: {train_duplicates}\n"
            f"[Test Set] {len(overlap_with_train)} out of {len(test_prompts)} are also in the train set.",
        )

    def get_model(self) -> nn.Module:
        """Get the model."""
        max_seq_len = self.config.chain_length * self.config.num_chains * 4 + 2
        model = MultilayerTransformer(
            vocab_size=39,
            d_model=self.config.dim_model,
            n_heads=self.config.num_heads,
            n_layers=self.config.num_layers,
            max_seq_len=max_seq_len,
        )
        return model

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Returns a basic Adam optimizer."""
        return optim.Adam(model.parameters(), lr=1e-3)

    def get_loss(self, model: nn.Module, batch: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the cross-entropy loss over the final token logits.

        Example prompt and answer (where . is masked out):
            Prompt:  "|1>a|2>b|a>c|b>d"
            Mask:    "0101010101010101"
            Answers: " 1 1 2 2 1 1 2 2"
        """
        x = batch["prompt_tokens"]  # (batch, seq)
        y = batch["answer_tokens"]  # (batch, seq)
        mask = (y != MASK_ID).long()  # (batch, seq)

        logits, _, _ = model(x)  # (batch, seq, vocab_size)
        vocab_size = logits.shape[-1]

        # flatten for masked loss computation
        logits_flat = logits.view(-1, vocab_size)  # (batch*seq, vocab_size)
        y_flat = y.view(-1)  # (batch*seq)
        mask_flat = mask.view(-1).bool()  # (batch*seq)

        # get cross-entropy loss over masked positions
        logits_masked = logits_flat[mask_flat]  # (n, vocab_size)
        y_masked = y_flat[mask_flat]  # (n)
        loss = torch.nn.functional.cross_entropy(logits_masked, y_masked)

        return loss, logits

    def val_metrics(self, model: nn.Module, batch: TensorTree, preds: torch.Tensor) -> dict[str, float | str]:
        """Get additional validation metrics for a batch."""
        predicted_answers = preds.argmax(dim=-1)  # (batch, seq)
        target = batch["answer_tokens"].to(predicted_answers.device)  # (batch, seq)
        mask = target != MASK_ID  # (batch, seq)

        # accuracy only over masked positions
        correct = ((predicted_answers == target) & mask).float()
        total_accuracy = correct.sum() / mask.sum()

        # sample info for logging
        sample_idx = 0
        sample_prompt = batch["prompt"][sample_idx]
        sample_answer = batch["answer"][sample_idx]
        sample_pred_token_ids = predicted_answers[sample_idx].tolist()
        predicted_answer = self.prompter.detokenize(sample_pred_token_ids)
        predicted_answer = "".join(c if i % 2 else "." for i, c in enumerate(predicted_answer))
        correct = "".join(
            "." if sample_answer[i] == MASK_TOKEN else "✓" if sample_answer[i] == predicted_answer[i] else "✗"
            for i in range(len(sample_answer))
        )
        sample_logits = preds[sample_idx]

        # find the probability of each predicted token
        sample_probs = torch.softmax(sample_logits, dim=-1)  # (seq, vocab)
        sample_pred_probs = sample_probs[torch.arange(sample_probs.size(0)), sample_pred_token_ids]  # (seq,)

        return {
            "sample_prompt": sample_prompt,
            "sample_answer": sample_answer,
            "predicted_answer": predicted_answer,
            "correct": correct,
            "probability": sample_pred_probs.mean().item(),  # avg probability of predicted tokens
            "accuracy": total_accuracy.item(),
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
    # Example usage (config gets overridden by CLI args):
    # python -m examples.var_rename --num_layers=3 --overwrite_existing_checkpoints
    config = VariableRenamingConfig(
        num_layers=3,
        dim_model=128,
        num_heads=1,
        # train_size=10_000_000,
        train_size=10_000,
        test_size=10_000,
        num_chains=2,
        chain_length=16,
        num_epochs=1000,
        batch_size=1024,
        eval_every_n_samples=10_000,
        log_every_n_seconds=3,
        dataset_path="data/var_rename",
        tensorboard_logdir="logs/var_rename",
        save_every_n_seconds=100,
        overwrite_existing_checkpoints=True,
    )
    trainer = VariableRenamingTrainer(config)
    trainer.run()
