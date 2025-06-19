"""Prompt generation and dataset generation."""

import hashlib
import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, Hashable, TypeVar

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
from matplotlib.pylab import default_rng
from tqdm import tqdm


def get_field_type(v: Any) -> Any:
    """Get the type name of a value."""
    if isinstance(v, bool):
        return Value("bool")
    elif isinstance(v, int):
        return Value("int32")
    elif isinstance(v, float):
        return Value("float32")
    elif isinstance(v, str):
        return Value("string")
    elif isinstance(v, list) or isinstance(v, tuple):
        return Sequence(feature=get_field_type(v[0]))
    else:
        raise ValueError(f"Unsupported type: {type(v)}")


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

    def __init__(self, config: ConfigType, seed: int = 42):
        """Initialize the synthesizer with a configuration."""
        self.config = config
        self.base_seed = seed

    @abstractmethod
    def get_prompt(self, rng: np.random.Generator) -> tuple[str, str, dict[str, Any]]:
        """Generate a prompt and its expected answer.

        Returns:
            prompt: The prompt string.
            answer: The expected answer string.
            metadata: A dictionary of metadata about the prompt.
        """
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

    def generate_example(self, idx: int) -> tuple[str, str, dict[str, Any], list[int], list[int]]:
        """Generate a single example of prompt and answer.

        Returns:
            prompt: The prompt string.
            answer: The expected answer string.
            metadata: A dictionary of metadata about the prompt.
        """
        rng = default_rng(self.base_seed + idx)
        prompt, answer, metadata = self.get_prompt(rng)
        return prompt, answer, metadata, self.tokenize(prompt), self.tokenize(answer)

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
        metadatas: dict[str, list[Any]] = {}

        dataset_size = train_size + test_size
        with ProcessPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(self.generate_example, range(dataset_size)),
                    total=dataset_size,
                    desc="Generating dataset",
                )
            )

        for prompt, answer, metadata, prompt_token, answer_token in results:
            prompts.append(prompt)
            answers.append(answer)
            prompt_tokens.append(prompt_token)
            answer_tokens.append(answer_token)
            for k, v in metadata.items():
                if k not in metadatas:
                    metadatas[k] = []
                metadatas[k].append(v)

        info = DatasetInfo(
            description=str(self.config.data_config),
            features=Features(
                {
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "prompt_tokens": Sequence(feature=Value("int32")),
                    "answer_tokens": Sequence(feature=Value("int32")),
                    **{k: get_field_type(v) for k, v in metadata.items()},
                }
            ),
        )

        dataset = Dataset.from_dict(
            {
                "prompt": prompts,
                "answer": answers,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                **metadatas,
            },
            info=info,
        )

        splits = dataset.train_test_split(test_size=test_size, train_size=train_size, shuffle=True, seed=seed)
        splits.save_to_disk(dataset_path)
        return splits
