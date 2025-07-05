# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch-based framework for running quick attention-based experiments, with a focus on transformer models and synthetic prompt generation. The main use case is training transformer models on synthetic tasks like variable renaming and evaluation.

## Development Commands

### Setup and Installation
```bash
# Install the package in development mode
pip install -e .

# Install dependencies
pip install -r iluvattnshun/requirements.txt
```

### Code Quality
```bash
# Type checking
mypy iluvattnshun/

# Code formatting  
black iluvattnshun/ --line-length 120

# Both type checking and formatting are configured in pyproject.toml
```

### Running Examples
```bash
# Run variable renaming experiment
python -m examples.var_rename

# Run with custom parameters
python -m examples.var_rename --num_layers=4 --num_heads=8 --run_name=my_experiment

# Run other examples
python -m examples.shakespeare
python -m examples.var_eval
```

### Training Management
- Training logs and checkpoints are saved to `logs/` directory
- Each run creates a timestamped directory with:
  - TensorBoard logs (`events.out.tfevents.*`)
  - Configuration file (`run_*.yaml`)
  - Source code snapshot (`run_*.py`)
  - Model checkpoints (`ckpt_epoch_*.pt`)

## Architecture

### Core Components

**`iluvattnshun/nn.py`** - Neural network modules:
- `MultilayerTransformer`: Main transformer model with RoPE positional encoding
- `Attention`: Multi-head attention with KV caching support
- `TransformerLayer`: Single transformer layer with pre-LayerNorm
- `RotaryEmbedding`: Rotary positional embedding implementation

**`iluvattnshun/trainer.py`** - Training framework:
- `Trainer`: Abstract base class for training loops
- `TrainerConfig`: Configuration for training parameters
- Handles model checkpointing, logging, evaluation cycles

**`iluvattnshun/prompter.py`** - Synthetic data generation:
- `Prompter`: Abstract base class for prompt generation
- `PromptConfig`: Configuration for dataset parameters
- Supports HuggingFace datasets format with versioning

**`iluvattnshun/logger.py`** - Logging utilities:
- TensorBoard integration
- Configurable logging intervals and precision
- Text and metrics logging

### Key Design Patterns

1. **Configuration-driven**: All experiments use dataclass configs that extend base classes
2. **Reproducible datasets**: Prompt configs generate deterministic hashes for dataset versioning
3. **Modular training**: Trainer class can be extended for different tasks
4. **Attention visualization**: Models support returning attention weights for analysis

### Example Structure

The `examples/` directory contains complete experiments:
- `var_rename/`: Variable renaming task with attention visualization
- `shakespeare.py`: Character-level language modeling
- `var_eval.py`: Variable evaluation task
- `dag.py`: Directed acyclic graph experiments

Each example follows the pattern:
1. Define task-specific config extending `PromptConfig` and `TrainerConfig`
2. Implement `Prompter` subclass for data generation
3. Implement `Trainer` subclass with task-specific loss and metrics
4. Use command-line arguments for hyperparameter sweeps

### Visualization

The framework includes attention visualization tools:
- `iluvattnshun/viz.py`: Plotly-based attention heatmaps
- Examples show how to extract and visualize attention patterns
- Supports multi-layer, multi-head attention analysis

## Testing

Currently no formal test suite exists. Development relies on:
- MyPy for type checking
- Black for code formatting
- Running examples to validate functionality

## Hyperparameter Sweeps

The examples include shell scripts for running grid searches with tmux:
- Use `CUDA_VISIBLE_DEVICES` for GPU assignment
- Sweep results saved to separate timestamped directories
- See `examples/README.md` for detailed sweep instructions