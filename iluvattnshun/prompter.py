from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from datasets import Dataset


@dataclass
class PromptConfig:
    """Base configuration class for prompt generation."""

    num_prompts: int


T = TypeVar("T", bound=PromptConfig)


class Prompter(ABC, Generic[T]):
    """Base class for prompt generation.

    This abstract class defines the interface for prompt generators.
    Subclasses must implement get_prompt and tokenize methods.
    """

    def __init__(self, config: T):
        """Initialize the prompter with a configuration."""
        self.config = config

    @abstractmethod
    def get_prompt(self) -> tuple[str, str]:
        """Generate a prompt and its expected answer."""
        # TODO: think about implementing pseudorandomness nicely.
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

    def make_dataset(self, path: str) -> None:
        """Generate a HuggingFace dataset of prompts and answers with config.

        Args:
            path: Path to save the dataset to.
        """
        prompts = []
        answers = []
        prompt_tokens = []
        answer_tokens = []

        for _ in range(self.config.num_prompts):
            prompt, answer = self.get_prompt()
            prompts.append(prompt)
            answers.append(answer)
            prompt_tokens.append(self.tokenize(prompt))
            answer_tokens.append(self.tokenize(answer))

        dataset = Dataset.from_dict(
            {
                "prompt": prompts,
                "answer": answers,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
            }
        )

        dataset.save_to_disk(path)
