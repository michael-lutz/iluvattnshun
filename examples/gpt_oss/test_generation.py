"""Test text generation with the model."""
import sys
import torch
from pathlib import Path

from model.model import TokenGenerator
from model.tokenizer import get_tokenizer

_PATH_TO_CHECKPOINT = "/home/ubuntu/michael-base/iluvattnshun/weights/gpt-oss-20b/original"

_PROMPT = """
In the codebase below, a bug occurs. I know f11164() gets called. Which function caused the bug? Provide your answer and then why.

def f14698():
    raise RuntimeError('Unknown error occurred')

def f11164():
    raise RuntimeError('Unknown error occurred')

def f18424():
    raise RuntimeError('Unknown error occurred')

def f11437():
    raise RuntimeError('Unknown error occurred')

def f16675():
    f18424()

def f19706():
    f11437()

def f14318():
    f19706()

def f13588():
    f11164()

def f19781():
    f16675()

def f13578():
    f13588()

def f10571():
    f14698()

def f14037():
    f19781()

def f15394():
    f14037()

def f14642():
    f10571()

def f15294():
    f14642()

def f11955():
    f14318()

def f19783():
    f13578()

def f11098():
    f15394()

def f12421():
    f11955()

def f14363():
    f11098()

def f13261():
    f12421()

def f12227():
    f15294()

def f18920():
    f13261()

def f17094():
    f18920()

def f17819():
    f19783()

def f10747():
    f14363()

def f13086():
    f10747()
"""



def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else _PATH_TO_CHECKPOINT
    prompt = sys.argv[2] if len(sys.argv) > 2 else _PROMPT

    print("="*70)
    print("GPT-OSS-20B Text Generation Test")
    print("="*70)

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memory: {gpu_memory:.1f} GB")
        # Clear any cached memory
        torch.cuda.empty_cache()

    # Load tokenizer
    print(f"\n1. Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"   ✓ Tokenizer loaded")

    # Load model
    print(f"\n2. Loading model from checkpoint...")
    print(f"   Path: {checkpoint_path}")
    print(f"   This may take a few minutes...")
    generator = TokenGenerator(checkpoint_path, device=device)
    print(f"   ✓ Model loaded")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"   GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

    # Tokenize prompt
    print(f"\n3. Generating text...")
    print(f"   Prompt: \"{prompt}\"")
    prompt_tokens = tokenizer.encode(prompt)
    print(f"   Prompt tokens: {len(prompt_tokens)}")

    # Generate
    print(f"\n   Generated text:")
    print(f"   {prompt}", end="", flush=True)

    stop_tokens = [tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]]
    max_tokens = 50

    generated_tokens = []
    for token in generator.generate(
        prompt_tokens=prompt_tokens,
        stop_tokens=stop_tokens,
        temperature=0.8,
        max_tokens=max_tokens,
        return_logprobs=False
    ):
        generated_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)

    print("\n")
    print(f"\n   Generated {len(generated_tokens)} tokens")

    print("\n" + "="*70)
    print("✓ Generation complete!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
