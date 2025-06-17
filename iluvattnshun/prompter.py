"""Prompt generation and dataset generation."""

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Sequence,
    Value,
    load_from_disk,
)


@dataclass(kw_only=True)
class PromptConfig(ABC):
    """Base configuration class for prompt generation."""

    @abstractmethod
    def data_hash_params(self) -> list[str]:
        """List of parameters to include in the data hash."""
        pass

    @property
    def data_config(self) -> dict[str, int | float | str | bool]:
        """Dictionary of parameters to include in the data hash."""
        res = {}
        for k, v in self.__dict__.items():
            if k in self.data_hash_params():
                assert isinstance(v, int | float | str | bool), f"Parameter {k} is not a hashable type: {type(v)}"
                res[k] = v
        return res

    def __hash__(self) -> int:
        """Generate a hash of the config for dataset versioning."""
        return self.get_hash()

    def get_hash(self) -> int:
        """Generate a hash of the config for dataset versioning."""
        hashable_repr = ""
        for k, v in self.data_config.items():
            hashable_repr += f"{k}={v}\n"

        return int(hashlib.sha256(hashable_repr.encode()).hexdigest(), 16)


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

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str:
        """Detokenize a list of token IDs.

        Args:
            tokens: List of token IDs
        """
        pass

    @property
    def dataset_name(self) -> str:
        """Get the name of the dataset."""
        return f"{self.__class__.__name__}_{self.config.get_hash()}".lower()

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
        dataset_name = self.dataset_name
        dataset_path = os.path.join(path, dataset_name)

        # first check if the dataset already exists & matches the config
        if os.path.exists(dataset_path):
            ds = load_from_disk(dataset_path)
            assert isinstance(ds, DatasetDict)
            for split in ds.keys():
                if ds[split].info.description != str(self.config.data_config):
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
            description=str(self.config.data_config),
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
        splits.save_to_disk(dataset_path)
        return splits
