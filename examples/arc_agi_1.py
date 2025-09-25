"""ARC-AGI-1 training example with data augmentation and tokenization."""

import hashlib
import json
import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Constants
MAX_GRID_LEN = 30
PADDING_TOKEN = 0
EOS_TOKEN = 1


@dataclass(frozen=True)
class TrainingSample:
    """Standardized training sample for ARC-AGI tasks.

    Args:
        context: Tuple of (input, output) pairs showing the pattern
        query: Input grid for the test case
        answer: Expected output grid for the test case
    """

    context: Tuple[Tuple[np.ndarray, np.ndarray], ...]
    query: np.ndarray
    answer: np.ndarray


def parse_arc_file(data_path: str) -> TrainingSample:
    """Parse an entire ARC data file into a single TrainingSample object.

    Args:
        data_path: Path to JSON file containing ARC data

    Returns:
        TrainingSample object with context from train examples and query/answer from test
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    # Parse training examples as context
    train_examples = data.get("train", [])
    context = []
    for example in train_examples:
        input_grid = np.array(example["input"], dtype=np.int32)
        output_grid = np.array(example["output"], dtype=np.int32)
        context.append((input_grid, output_grid))

    # Parse test example as query/answer
    test_examples = data.get("test", [])
    if not test_examples:
        raise ValueError("No test examples found in ARC file")

    test_example = test_examples[0]  # Take first test example
    query = np.array(test_example["input"], dtype=np.int32)
    answer = np.array(test_example["output"], dtype=np.int32)

    return TrainingSample(context=tuple(context), query=query, answer=answer)


# Abstract base class for grid augmentations
class GridAugment(ABC):
    """Abstract base class for grid augmentations."""

    @abstractmethod
    def augment(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply augmentation to input grid.

        Args:
            input_grid: Input grid to augment

        Returns:
            Augmented grid
        """
        pass


@dataclass(frozen=True)
class TranslateAug(GridAugment):
    """Translation augmentation that places input in random position within max grid size."""

    max_size: int = MAX_GRID_LEN
    padding_value: int = -1

    def augment(self, input_grid: np.ndarray) -> np.ndarray:
        """Translate grid to random position within max_size x max_size grid."""
        h, w = input_grid.shape

        # Create output grid filled with padding
        output = np.full((self.max_size, self.max_size), self.padding_value, dtype=np.int32)

        # Random position for top-left corner
        max_row = max(0, self.max_size - h)
        max_col = max(0, self.max_size - w)

        if max_row > 0:
            start_row = random.randint(0, max_row)
        else:
            start_row = 0

        if max_col > 0:
            start_col = random.randint(0, max_col)
        else:
            start_col = 0

        # Place input grid in output
        end_row = start_row + h
        end_col = start_col + w
        output[start_row:end_row, start_col:end_col] = input_grid

        return output


@dataclass(frozen=True)
class RotateAug(GridAugment):
    """Rotation augmentation in 90-degree increments."""

    def augment(self, input_grid: np.ndarray) -> np.ndarray:
        """Rotate grid by random multiple of 90 degrees."""
        rotations = random.randint(0, 3)
        return np.rot90(input_grid, k=rotations)


@dataclass(frozen=True)
class ReflectAug(GridAugment):
    """Reflection augmentation across x or y axes."""

    def augment(self, input_grid: np.ndarray) -> np.ndarray:
        """Reflect grid across random axis."""
        if random.random() < 0.5:
            return np.fliplr(input_grid)  # reflect across y-axis
        else:
            return np.flipud(input_grid)  # reflect across x-axis


@dataclass(frozen=True)
class AugmentSequence:
    """Sequence of augmentations to apply in order."""

    augmentations: Tuple[GridAugment, ...]

    def augment(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply all augmentations in sequence."""
        result = input_grid
        for aug in self.augmentations:
            result = aug.augment(result)
        return result


def create_random_augment_sequence() -> AugmentSequence:
    """Create a random sequence of augmentations."""
    # Always include translation to ensure consistent sizing
    augs: List[GridAugment] = [TranslateAug()]

    # Randomly add rotation and reflection
    if random.random() < 0.5:
        augs.append(RotateAug())
    if random.random() < 0.5:
        augs.append(ReflectAug())

    # Shuffle order (but translation should be last for consistent sizing)
    if len(augs) > 1:
        translate_aug = augs.pop(0)
        random.shuffle(augs)
        augs.append(translate_aug)

    return AugmentSequence(tuple(augs))


def augment_training_sample(sample: TrainingSample, augment_sequence: AugmentSequence) -> TrainingSample:
    """Apply augmentation sequence to a training sample."""
    # Augment context examples
    augmented_context = []
    for input_grid, output_grid in sample.context:
        aug_input = augment_sequence.augment(input_grid)
        aug_output = augment_sequence.augment(output_grid)
        augmented_context.append((aug_input, aug_output))

    # Augment query and answer
    aug_query = augment_sequence.augment(sample.query)
    aug_answer = augment_sequence.augment(sample.answer)

    return TrainingSample(context=tuple(augmented_context), query=aug_query, answer=aug_answer)


def add_eos_tokens(grid: np.ndarray) -> np.ndarray:
    """Add end-of-sequence tokens to right and bottom edges of grid."""
    h, w = grid.shape
    # Create output with EOS tokens on right and bottom
    output = np.full((h + 1, w + 1), EOS_TOKEN, dtype=np.int32)
    output[:h, :w] = grid
    return output


def tokenize_training_sample(sample: TrainingSample) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize a training sample into flattened token sequences.

    Args:
        sample: TrainingSample to tokenize

    Returns:
        Tuple of (input_tokens, output_tokens) as flattened arrays
    """
    input_tokens = []

    # Process context examples
    for input_grid, output_grid in sample.context:
        # Add EOS tokens to input
        input_with_eos = add_eos_tokens(input_grid)
        input_tokens.extend(input_with_eos.flatten())

        # Add EOS tokens to output (but not bottom row for output)
        output_with_eos = np.full((output_grid.shape[0] + 1, output_grid.shape[1] + 1), EOS_TOKEN, dtype=np.int32)
        output_with_eos[:-1, :-1] = output_grid  # Don't add bottom row EOS
        input_tokens.extend(output_with_eos.flatten())

    # Process query
    query_with_eos = add_eos_tokens(sample.query)
    input_tokens.extend(query_with_eos.flatten())

    # Process answer (without bottom row EOS)
    answer_with_eos = np.full((sample.answer.shape[0] + 1, sample.answer.shape[1] + 1), EOS_TOKEN, dtype=np.int32)
    answer_with_eos[:-1, :-1] = sample.answer  # Don't add bottom row EOS
    output_tokens = answer_with_eos.flatten()

    return np.array(input_tokens, dtype=np.int32), np.array(output_tokens, dtype=np.int32)


def generate_dataset_hash(samples: List[TrainingSample], num_augmentations: int) -> str:
    """Generate a hash for the dataset based on samples and augmentation count."""
    # Create a string representation of the dataset
    dataset_str = f"num_samples:{len(samples)},num_augmentations:{num_augmentations}"

    # Add sample information
    for i, sample in enumerate(samples[:5]):  # Use first 5 samples for hash
        dataset_str += f",sample_{i}_context_len:{len(sample.context)}"
        dataset_str += f",sample_{i}_query_shape:{sample.query.shape}"
        dataset_str += f",sample_{i}_answer_shape:{sample.answer.shape}"

    return hashlib.md5(dataset_str.encode()).hexdigest()[:12]


def generate_augmented_dataset(base_sample: TrainingSample, num_augmentations: int, output_dir: str) -> str:
    """Generate augmented dataset and save to disk.

    Args:
        base_sample: Base training sample to augment
        num_augmentations: Number of augmented versions to create
        output_dir: Directory to save the dataset

    Returns:
        Path to the saved dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate dataset hash
    dataset_hash = generate_dataset_hash([base_sample], num_augmentations)

    # Create augmented samples
    augmented_samples = []
    for _ in range(num_augmentations):
        augment_sequence = create_random_augment_sequence()
        augmented_sample = augment_training_sample(base_sample, augment_sequence)
        augmented_samples.append(augmented_sample)

    # Tokenize all samples
    tokenized_data = []
    for sample in augmented_samples:
        input_tokens, output_tokens = tokenize_training_sample(sample)
        tokenized_data.append({"input_tokens": input_tokens.tolist(), "output_tokens": output_tokens.tolist()})

    # Save dataset
    dataset_path = os.path.join(output_dir, f"arc_agi_1_{dataset_hash}.json")
    with open(dataset_path, "w") as f:
        json.dump(
            {
                "dataset_hash": dataset_hash,
                "num_base_samples": 1,
                "num_augmentations": num_augmentations,
                "total_samples": len(tokenized_data),
                "data": tokenized_data,
            },
            f,
            indent=2,
        )

    return dataset_path


class ArcDataset(Dataset):
    """PyTorch dataset for ARC-AGI-1 data."""

    def __init__(self, dataset_path: str):
        """Initialize dataset from saved JSON file."""
        with open(dataset_path, "r") as f:
            data = json.load(f)

        self.data = data["data"]
        self.dataset_hash = data["dataset_hash"]
        self.num_base_samples = data["num_base_samples"]
        self.num_augmentations = data["num_augmentations"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.data[idx]
        input_tokens = torch.tensor(sample["input_tokens"], dtype=torch.long)
        output_tokens = torch.tensor(sample["output_tokens"], dtype=torch.long)
        return input_tokens, output_tokens


def create_dataloader(
    dataset_path: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    """Create a PyTorch DataLoader for the ARC dataset."""
    dataset = ArcDataset(dataset_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: (torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch])),
    )


def main():
    """Main function to test the ARC-AGI-1 implementation."""
    print("Testing ARC-AGI-1 implementation...")

    # Test data loading
    print("\n1. Testing data loading...")
    data_path = "/home/michael-lutz/iluvattnshun/data/arc_agi_1/data/training/0a938d79.json"
    if os.path.exists(data_path):
        sample = parse_arc_file(data_path)
        print(f"Loaded sample from {data_path}")
        print(f"Context length: {len(sample.context)}")
        print(f"Query shape: {sample.query.shape}")
        print(f"Answer shape: {sample.answer.shape}")

        # Show context examples
        for i, (ctx_input, ctx_output) in enumerate(sample.context):
            print(f"  Context {i}: input {ctx_input.shape}, output {ctx_output.shape}")
    else:
        print(f"Data file not found: {data_path}")
        return

    # Test augmentations
    print("\n2. Testing augmentations...")
    test_grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    print(f"Original grid:\n{test_grid}")

    # Test translation
    translate_aug = TranslateAug(max_size=5)
    translated = translate_aug.augment(test_grid)
    print(f"Translated grid:\n{translated}")

    # Test rotation
    rotate_aug = RotateAug()
    rotated = rotate_aug.augment(test_grid)
    print(f"Rotated grid:\n{rotated}")

    # Test reflection
    reflect_aug = ReflectAug()
    reflected = reflect_aug.augment(test_grid)
    print(f"Reflected grid:\n{reflected}")

    # Test augmentation sequence
    print("\n3. Testing augmentation sequence...")
    augment_sequence = create_random_augment_sequence()
    augmented_sample = augment_training_sample(sample, augment_sequence)
    print(f"Augmented query shape: {augmented_sample.query.shape}")
    print(f"Augmented answer shape: {augmented_sample.answer.shape}")

    # Test tokenization
    print("\n4. Testing tokenization...")
    input_tokens, output_tokens = tokenize_training_sample(augmented_sample)
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Output tokens shape: {output_tokens.shape}")
    print(f"Input tokens (first 20): {input_tokens[:20]}")
    print(f"Output tokens (first 20): {output_tokens[:20]}")

    # Test dataset generation
    print("\n5. Testing dataset generation...")
    output_dir = "/tmp/arc_agi_1_test"
    dataset_path = generate_augmented_dataset(sample, 3, output_dir)
    print(f"Generated dataset at: {dataset_path}")

    # Test dataloader
    print("\n6. Testing dataloader...")
    dataloader = create_dataloader(dataset_path, batch_size=2, shuffle=False)
    print(f"Dataloader length: {len(dataloader)}")

    for i, (input_batch, output_batch) in enumerate(dataloader):
        print(f"Batch {i}: input shape {input_batch.shape}, output shape {output_batch.shape}")
        if i >= 2:  # Only show first few batches
            break

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
