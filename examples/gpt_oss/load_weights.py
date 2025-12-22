"""Script to load real gpt-oss-20b weights and configuration."""
import json
import sys
from pathlib import Path
import torch

from model.model import ModelConfig, Transformer
from model.weights import Checkpoint
from model.tokenizer import get_tokenizer

def load_config(checkpoint_path: str) -> ModelConfig:
    """Load ModelConfig from checkpoint directory."""
    config_file = Path(checkpoint_path) / "config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    config = ModelConfig(**config_dict)
    return config

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "../../weights/gpt-oss-20b/original"
    
    print("="*70)
    print("Loading GPT-OSS-20B Configuration")
    print("="*70)
    
    # Load config
    print(f"\n1. Loading config from: {checkpoint_path}")
    try:
        config = load_config(checkpoint_path)
        print("   ✓ Config loaded successfully!")
        print(f"\n   Model Configuration:")
        print(f"     - Layers: {config.num_hidden_layers}")
        print(f"     - Hidden size: {config.hidden_size}")
        print(f"     - Attention heads: {config.num_attention_heads}")
        print(f"     - KV heads: {config.num_key_value_heads}")
        print(f"     - Experts: {config.num_experts}")
        print(f"     - Experts per token: {config.experts_per_token}")
        print(f"     - Vocab size: {config.vocab_size}")
        print(f"     - Sliding window: {config.sliding_window}")
        print(f"     - RoPE theta: {config.rope_theta}")
    except Exception as e:
        print(f"   ✗ Failed to load config: {e}")
        return 1
    
    # Check for weight files
    print(f"\n2. Checking for weight files...")
    checkpoint_dir = Path(checkpoint_path)
    safetensor_files = list(checkpoint_dir.glob("*.safetensors"))
    
    if safetensor_files:
        print(f"   ✓ Found {len(safetensor_files)} safetensors files:")
        for f in sorted(safetensor_files)[:5]:
            size_mb = f.stat().st_size / (1024**2)
            print(f"     - {f.name} ({size_mb:.1f} MB)")
        if len(safetensor_files) > 5:
            print(f"     ... and {len(safetensor_files) - 5} more")
    else:
        print(f"   ⚠ No safetensors files found in {checkpoint_path}")
        print(f"   To download weights, run:")
        print(f"     python -c \"from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-20b', allow_patterns='original/*', local_dir='gpt-oss-20b')\"")
        return 0
    
    # Try loading checkpoint (if weights exist)
    print(f"\n3. Initializing Checkpoint loader...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = Checkpoint(checkpoint_path, device)
        print(f"   ✓ Checkpoint initialized on {device}")
        print(f"   ✓ Found {len(checkpoint.tensor_name_to_file)} tensors")
        
        # Show some example tensor names
        tensor_names = list(checkpoint.tensor_name_to_file.keys())
        print(f"\n   Sample tensor names:")
        for name in sorted(tensor_names)[:10]:
            print(f"     - {name}")
        if len(tensor_names) > 10:
            print(f"     ... and {len(tensor_names) - 10} more")
            
    except Exception as e:
        print(f"   ✗ Failed to initialize checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("✓ Successfully loaded config and checkpoint!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
