import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from datasets import Dataset, DatasetInfo, Features, Sequence, Value


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

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        pass

    def make_dataset(self, path: str, size: int, seed: int = 42) -> None:
        """Generate a HuggingFace dataset of prompts and answers with config.

        Args:
            path: Path to save the dataset to.
            seed: Seed for the random number generator.
        """
        prompts = []
        answers = []
        prompt_tokens = []
        answer_tokens = []
        np.random.seed(seed)  # If ever do multiproc, rethink...

        for _ in range(size):
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
        dataset.save_to_disk(path)
