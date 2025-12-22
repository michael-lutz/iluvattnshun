# GPT-OSS Model Files

This directory contains the essential PyTorch files from gpt-oss for loading and working with the model.

## Directory Structure

```
examples/gpt_oss/
├── model/               # Flat directory with all model files
│   ├── __init__.py
│   ├── model.py         # ModelConfig, RMSNorm, RotaryEmbedding, Transformer
│   ├── weights.py       # Checkpoint loading with MXFP4 support
│   ├── utils.py         # Utility functions
│   └── tokenizer.py     # Tokenizer using tiktoken
├── test_model.py        # Test script
└── README.md            # This file
```

## Files Copied

### model.py (16K)
- **ModelConfig**: Configuration dataclass for model architecture
- **RMSNorm**: RMS normalization layer
- **RotaryEmbedding**: Rotary positional embeddings
- **AttentionBlock**, **MLPBlock**, **TransformerBlock**: Model components
- **Transformer**: Full transformer model (reference PyTorch implementation)

### weights.py (5.1K)
- **Checkpoint**: Loads model weights from safetensors
- **MXFP4 support**: Handles MXFP4 quantized weights for MoE layers

### utils.py (1.2K)
- Utility functions for model operations

### tokenizer.py (1K)
- **get_tokenizer()**: Returns tiktoken tokenizer for gpt-oss

## Usage

### Basic Import
```python
from model.model import ModelConfig, RMSNorm
from model.weights import Checkpoint
from model.tokenizer import get_tokenizer

# Create config
config = ModelConfig()

# Use RMSNorm
norm = RMSNorm(config.hidden_size)

# Get tokenizer
tokenizer = get_tokenizer()
text_tokens = tokenizer.encode("Hello!")
```

### Test the Setup
```bash
python test_model.py
```

Expected output:
- ✓ ModelConfig created
- ✓ RMSNorm forward pass works  
- ✓ Tokenizer works

## Dependencies

- **torch**: PyTorch (already installed)
- **safetensors**: For loading weights (already installed)
- **tiktoken**: For tokenization (already installed)

## What's NOT Included

This is a minimal copy with only PyTorch/torch files. Not included:
- Triton kernels (triton/)
- Tools (browser, python)
- Evaluation suite (evals/)
- API servers (responses_api/)
- Other backends (metal/, vllm/)

This keeps the example minimal and focused on the core model components.

## Next Steps

To use these files for inference:
1. Download model weights from Hugging Face
2. Use Checkpoint to load weights
3. Create Transformer model
4. Run forward pass

For optimized inference with Triton kernels, see the main gpt-oss repository.

## Source

Files copied from: https://github.com/openai/gpt-oss
License: Apache 2.0
