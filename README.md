# I ♥ Attention
This is a general repository for running quick attention-based experiments.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Prerequisites
- Python 3.11 (3.12 not yet supported)
- NVIDIA GPU with CUDA support (tested on GH200 with CUDA 12.x)
- uv package manager

### Setup

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Sync project dependencies**:
```bash
# This installs all dependencies EXCEPT PyTorch
uv sync
```

3. **Install PyTorch with CUDA support**:

PyTorch is intentionally NOT managed by uv because CUDA installations are system-specific.

```bash
# Activate the virtual environment
source .venv/bin/activate

# For x86_64 with CUDA 12.4
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For ARM64/aarch64 (like GH200), PyTorch CUDA wheels aren't available from PyTorch.org
# Option A: Use system-installed PyTorch by setting PYTHONPATH (if using Python 3.10)
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

# Option B: Build from source (see https://github.com/pytorch/pytorch#from-source)
# Option C: Use pre-built system package if available
```

4. **Verify GPU setup**:
```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
```

You should see `CUDA: True` and your GPU name.

### Running Scripts

When running scripts, ensure PyTorch is accessible:

```bash
source .venv/bin/activate

# If torch is in venv, just run normally
python script.py

# If using system torch (Python 3.10), set PYTHONPATH
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH python script.py
```

## Quick Start

See [CLAUDE.md](CLAUDE.md) for detailed development commands and examples.

## Example-Specific Requirements

Some examples have additional dependencies beyond the base installation:

- **GPT-OSS example** (`examples/gpt_oss/`): See [examples/gpt_oss/README.md](examples/gpt_oss/README.md)
