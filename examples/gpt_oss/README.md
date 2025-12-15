# Triton Implementation for Shakespeare Training

This directory contains a triton-optimized transformer implementation adapted from gpt-oss for training on the Shakespeare dataset.

## Files

- `config.py`: Simplified ModelConfig and RMSNorm for shakespeare training
- `attention.py`: FlashAttention implementation with learned sinks and banded attention
- `moe.py`: MoE implementation with fallback for when triton_kernels is not available
- `model.py`: Transformer model using triton attention and MoE
- `wrapper.py`: Wrapper to make triton Transformer compatible with shakespeare trainer
- `shakespeare.py`: Training script for shakespeare dataset

## Usage

Run the training script:

```bash
python examples/triton/shakespeare.py
```

## Dependencies

- `torch`: PyTorch
- `triton`: Triton compiler (required for attention kernels)
- `tensorflow_datasets`: For loading shakespeare dataset
- `triton_kernels` (optional): For optimized MoE kernels. If not available, falls back to simplified implementation.

Note: Triton is required. Install it with:
```bash
pip install triton
```

## Notes

- The implementation works without `triton_kernels` by using a simplified MoE implementation
- For single-expert models (num_experts=1), the MoE routing is bypassed
- The model uses bfloat16 for weights and activations
