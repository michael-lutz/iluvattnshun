"""Test loading the full model and running a simple forward pass."""
import json
import sys
import time
import torch
from pathlib import Path

from model.model import ModelConfig, Transformer
from model.tokenizer import get_tokenizer

_PATH_TO_CHECKPOINT = "/home/ubuntu/michael-base/iluvattnshun/weights/gpt-oss-20b/original"

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else _PATH_TO_CHECKPOINT
    
    print("="*70)
    print("GPT-OSS-20B Model Loading and Inference Test")
    print("="*70)
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load config
    print(f"\n1. Loading configuration...")
    config_file = Path(checkpoint_path) / "config.json"
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    config = ModelConfig(**config_dict)
    print(f"   ✓ Config loaded")
    print(f"     - {config.num_hidden_layers} layers")
    print(f"     - {config.num_experts} experts ({config.experts_per_token} active)")
    
    # Load model
    print(f"\n2. Loading model from checkpoint...")
    print(f"   This may take a few minutes...")
    start_time = time.time()
    
    try:
        # from_checkpoint reads config from checkpoint_path/config.json automatically
        model = Transformer.from_checkpoint(checkpoint_path, device=device)
        load_time = time.time() - start_time
        print(f"   ✓ Model loaded in {load_time:.1f}s")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Total parameters: {total_params / 1e9:.2f}B")
        
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test tokenizer
    print(f"\n3. Testing tokenizer...")
    tokenizer = get_tokenizer()
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    print(f"   ✓ Tokenized: '{test_text}' -> {len(tokens)} tokens")
    
    # Test forward pass
    print(f"\n4. Running forward pass...")
    try:
        # Create dummy input
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        print(f"   Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            start_time = time.time()
            logits = model(input_ids)
            forward_time = time.time() - start_time
        
        print(f"   ✓ Forward pass completed in {forward_time*1000:.1f}ms")
        print(f"   Output shape: {logits.shape}")
        print(f"   Output dtype: {logits.dtype}")
        
        # Get top predictions
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)
        
        print(f"\n   Top {top_k} predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token_text = tokenizer.decode([idx.item()])
            print(f"     {i+1}. '{token_text}' (prob: {prob.item():.4f})")
        
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
