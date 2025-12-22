"""Test text generation with CPU-offloaded experts to avoid OOM."""
import sys
import gc
import torch
from pathlib import Path

from model.model_offloaded import TransformerOffloaded
from model.tokenizer import get_tokenizer

_PATH_TO_CHECKPOINT = "/home/ubuntu/michael-base/iluvattnshun/weights/gpt-oss-20b/original"

# Full prompt - use for testing
_PROMPT_FULL = """
<|start|>system
model_identity: GPT-OSS
reasoning_effort: high
<|end|>

<|start|>developer
Write a function that causes a bug in the codebase.
<|end|>

<|start|>user
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
<|end|>
"""

# Shorter prompt to reduce activation memory
_PROMPT_SHORT = """<|start|>user
Write a Python function to calculate fibonacci numbers.
<|end|>
<|start|>assistant
"""

# Use full prompt by default to test user's scenario
_PROMPT = _PROMPT_FULL


class TokenGeneratorOffloaded:
    """Generator using offloaded model."""

    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = TransformerOffloaded.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            input_ids = torch.as_tensor([tokens], dtype=torch.int32, device=self.device)
            logits = self.model(input_ids)[0, -1]
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else _PATH_TO_CHECKPOINT
    prompt = sys.argv[2] if len(sys.argv) > 2 else _PROMPT

    print("="*70)
    print("GPT-OSS-20B Text Generation (CPU-Offloaded Experts)")
    print("="*70)

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memory: {gpu_memory:.1f} GB")
        torch.cuda.empty_cache()

    # Load tokenizer
    print(f"\n1. Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"   ✓ Tokenizer loaded")

    # Load model with offloading
    print(f"\n2. Loading model with expert offloading...")
    print(f"   Path: {checkpoint_path}")
    print(f"   Experts will be kept on CPU, moved to GPU on-demand")
    print(f"   This may take a few minutes...")
    generator = TokenGeneratorOffloaded(checkpoint_path, device=device)
    print(f"   ✓ Model loaded")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"   GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
        print(f"   (Compare to ~214 GB without offloading!)")

    # Tokenize prompt
    print(f"\n3. Generating text...")
    # Allow special tokens in the prompt
    prompt_tokens = tokenizer.encode(prompt, allowed_special="all")
    print(f"   Prompt tokens: {len(prompt_tokens)}")

    # Severely limit generation due to quadratic activation memory growth
    # With long prompts (>300 tokens), even 3 tokens may OOM!
    max_tokens = 3 if len(prompt_tokens) > 100 else 5
    print(f"   Max tokens to generate: {max_tokens}")
    print(f"   Note: Quadratic memory growth - each token reprocesses full context")
    print(f"   WARNING: Long prompts may still OOM due to lack of KV caching")

    # Generate
    print(f"\n   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"\n   Generated text:", end="", flush=True)

    stop_tokens = [tokenizer.encode("<|endoftext|>", allowed_special="all")[0]]

    generated_tokens = []
    try:
        for i, token in enumerate(generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            temperature=0.8,
            max_tokens=max_tokens,
            return_logprobs=False
        )):
            generated_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)

            # Aggressive memory cleanup after each token
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated(0) / 1e9
                print(f" [{mem:.1f}GB]", end="", flush=True)
                torch.cuda.empty_cache()
                gc.collect()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n\n   ERROR: GPU out of memory after {len(generated_tokens)} tokens")
            print(f"   Try reducing prompt length or max_tokens")
            if torch.cuda.is_available():
                print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        raise

    print("\n")
    print(f"\n   Generated {len(generated_tokens)} tokens")

    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   Final GPU memory: {final_memory:.2f} GB")

    print("\n" + "="*70)
    print("✓ Generation complete!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
