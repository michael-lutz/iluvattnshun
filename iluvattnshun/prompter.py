import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Sequence,
    Value,
    load_dataset,
)


@dataclass(kw_only=True)
class PromptConfig:
    """Base configuration class for prompt generation."""

    def __str__(self) -> str:
        """String representation of the config."""
        items = sorted(self.__dict__.items())
        res = "{"
        for k, v in items:
            res += f"\t{k}={v}\n"
        res += "}"
        return res

    def __hash__(self) -> int:
        """Generate a hash of the config for dataset versioning."""
        return int(hashlib.sha256(str(self).encode()).hexdigest(), 16)


ConfigType = TypeVar("ConfigType", bound=PromptConfig)


class Prompter(ABC, Generic[ConfigType]):
    """Base class for prompt generation.

    This abstract class defines the interface for prompt generators.
    Subclasses must implement get_prompt and tokenize methods.
    """

    def __init__(self, config: ConfigType):
        """Initialize the synthesizer with a configuration."""
        self.config = config

    @abstractmethod
    def get_prompt(self) -> tuple[str, str]:
        """Generate a prompt and its expected answer."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Tokenize an input text string.

        Args:
            text: Any-length text input.

        Returns:
            List of token IDs
        """
        pass

    def make_dataset(self, path: str, train_size: int, test_size: int, seed: int = 42) -> DatasetDict:
        """Generate a HuggingFace dataset of prompts and answers with config.

        Args:
            path: Path to save the dataset to.
            train_size: Number of training examples to generate.
            test_size: Number of test examples to generate.
            seed: Seed for the random number generator.

        Returns:
            The generated dataset.
        """

        # first check if the dataset already exists & matches the config
        if os.path.exists(path):
            ds = load_dataset(path)
            assert isinstance(ds, DatasetDict)
            for split in ds.keys():
                if ds[split].info.description != str(self.config):
                    break
            else:
                return ds

        prompts = []
        answers = []
        prompt_tokens = []
        answer_tokens = []
        np.random.seed(seed)  # if ever do multiproc, rethink...

        for _ in range(train_size + test_size):
            prompt, answer = self.get_prompt()
            prompts.append(prompt)
            answers.append(answer)
            prompt_tokens.append(self.tokenize(prompt))
            answer_tokens.append(self.tokenize(answer))

        info = DatasetInfo(
            description=str(self.config),
            features=Features(
                {
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "prompt_tokens": Sequence(feature=Value("int32")),
                    "answer_tokens": Sequence(feature=Value("int32")),
                }
            ),
        )

        dataset = Dataset.from_dict(
            {
                "prompt": prompts,
                "answer": answers,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
            },
            info=info,
        )

        splits = dataset.train_test_split(test_size=test_size, train_size=train_size, shuffle=True, seed=seed)
        splits.save_to_disk(path)
        return splits
