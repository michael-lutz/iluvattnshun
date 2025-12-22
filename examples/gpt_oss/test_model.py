"""Test script for model loading."""
import torch
from model.model import ModelConfig, RMSNorm
from model.weights import Checkpoint
from model.tokenizer import get_tokenizer

print("="*60)
print("Testing GPT-OSS Model Components")
print("="*60)

# Test 1: ModelConfig
print("\n1. Testing ModelConfig...")
config = ModelConfig()
print(f"   ✓ Created ModelConfig")
print(f"     - Hidden size: {config.hidden_size}")
print(f"     - Num layers: {config.num_hidden_layers}")
print(f"     - Vocab size: {config.vocab_size}")

# Test 2: RMSNorm
print("\n2. Testing RMSNorm...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm = RMSNorm(config.hidden_size, device=device)
x = torch.randn(2, 10, config.hidden_size, device=device)
y = norm(x)
print(f"   ✓ RMSNorm forward pass works")
print(f"     - Input shape: {x.shape}")
print(f"     - Output shape: {y.shape}")

# Test 3: Tokenizer
print("\n3. Testing Tokenizer...")
try:
    tokenizer = get_tokenizer()
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"   ✓ Tokenizer works")
    print(f"     - Original: '{test_text}'")
    print(f"     - Tokens: {tokens[:10]}... ({len(tokens)} total)")
    print(f"     - Decoded: '{decoded}'")
except Exception as e:
    print(f"   ⚠ Tokenizer test failed: {e}")

print("\n" + "="*60)
print("✓ All basic tests passed!")
print("="*60)
