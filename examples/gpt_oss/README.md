# GPT-OSS-20B Example

This example demonstrates how to use the GPT-OSS-20B model for inference and training on custom datasets.

## Additional Requirements

Beyond the base `iluvattnshun` dependencies, this example requires:

```bash
# Activate your environment first
source ../../.venv/bin/activate

# Install additional dependencies
uv pip install tiktoken tensorflow-datasets
```

## Model Weights

Download the GPT-OSS-20B model weights:

```bash
# Create weights directory
mkdir -p ../../weights/gpt-oss-20b

# Download weights (follow instructions from model repository)
# Expected structure: ../../weights/gpt-oss-20b/original/
```

## Running Inference

```bash
# Test basic model loading and generation
PYTHONPATH=/lambda/nfs/michael-base/iluvattnshun:$PYTHONPATH python test_generation.py

# With custom prompt
PYTHONPATH=/lambda/nfs/michael-base/iluvattnshun:$PYTHONPATH python test_generation.py "../../weights/gpt-oss-20b/original" "Your prompt here"
```

## Training on Shakespeare

The `train_shakespeare.py` script fine-tunes GPT-OSS-20B on the Tiny Shakespeare dataset using the iluvattnshun trainer API.

### Full Fine-tune (Default)

```bash
# Train with tiny batch size for memory efficiency
PYTHONPATH=/lambda/nfs/michael-base/iluvattnshun:$PYTHONPATH python train_shakespeare.py
```

### Frozen Layers + New Layer

Train only a new transformer layer added to the end while freezing the base model:

```bash
# Modify the config in train_shakespeare.py:
# config = GPTShakespeareConfig(
#     ...
#     freeze_base_model=True,
#     add_extra_layer=True,
#     ...
# )

PYTHONPATH=/lambda/nfs/michael-base/iluvattnshun:$PYTHONPATH python train_shakespeare.py
```

### Configuration

Key training parameters in `train_shakespeare.py`:

- `batch_size`: Start with 1 for 20B model
- `max_context_length`: Token sequence length (default: 256)
- `learning_rate`: Default 1e-5 for fine-tuning
- `freeze_base_model`: Set to `True` to only train new layers
- `add_extra_layer`: Set to `True` to add a new trainable layer
- `eval_every_n_samples`: How often to run evaluation
- `save_every_n_seconds`: Checkpoint frequency

### Monitoring Training

Logs and checkpoints are saved to `logs/gpt_shakespeare/`:

```bash
# View training with TensorBoard
tensorboard --logdir logs/gpt_shakespeare
```

## Files

- `model/model.py`: GPT-OSS-20B model architecture
- `model/tokenizer.py`: O200K tokenizer
- `model/weights.py`: Weight loading utilities
- `test_generation.py`: Inference testing script
- `test_inference.py`: Model loading test
- `train_shakespeare.py`: Shakespeare fine-tuning with iluvattnshun trainer
- `test_train_setup.py`: Training setup validation

## Notes

- The 20B model requires significant GPU memory. Start with `batch_size=1`.
- For faster experimentation, use the frozen layers + new layer paradigm which only trains ~0.05% of parameters.
- The model uses bfloat16 precision by default for memory efficiency.
- Training checkpoints include model state, optimizer state, and metrics for resuming.
