"""Wrapper to make triton Transformer compatible with shakespeare trainer."""

import torch
import torch.nn as nn

from .model import Transformer, Cache
from .config import ModelConfig


class TritonTokenTransformer(nn.Module):
    """Wrapper for triton Transformer to match TokenTransformer interface."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 8,
        rope_base: float = 1024.0,
        max_context_length: int = 128,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_context_length = max_context_length

        # create config
        head_dim = d_model // n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        config = ModelConfig(
            num_hidden_layers=n_layers,
            num_experts=1,
            experts_per_token=1,
            vocab_size=vocab_size,
            hidden_size=d_model,
            intermediate_size=4 * d_model,
            head_dim=head_dim,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            sliding_window=0,
            initial_context_length=max_context_length,
            rope_theta=rope_base,
            rope_scaling_factor=1.0,
        )

        self.model = Transformer(config=config, device=device)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_attn_weights: bool = False,
        return_xs: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Forward pass matching TokenTransformer interface.

        Args:
            x: Input tokens of shape (batch_size, seq_len)
            kv_cache: Optional list of (key, value) tuples for each layer
            return_attn_weights: Whether to return attention weights
            return_xs: Whether to return intermediate x values

        Returns:
            Tuple of (logits, new_kv_cache, attention_weights, intermediate_xs)
        """
        batch_size, seq_len = x.shape

        # for training, we don't use kv_cache (caches=None)
        # forward pass
        logits = self.model(x, caches=None)

        # return empty kv_cache for compatibility
        new_kv_cache = []

        # dummy returns for compatibility
        attn_weights = [] if return_attn_weights else []
        xs = [] if return_xs else []

        return logits, new_kv_cache, attn_weights, xs

    def sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample token from logits."""
        next_token_logits = logits / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            prompt: Initial prompt tensor of shape (batch_size, prompt_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated sequence including prompt
        """
        batch_size = prompt.shape[0]
        device = prompt.device

        # create caches for generation
        caches = [
            Cache(batch_size, self.max_context_length, self.model.config.num_key_value_heads, device=device)
            for _ in range(self.model.config.num_hidden_layers)
        ]

        # run once on the whole prompt to prime the cache
        self.model(prompt, caches=caches)
        generated = prompt.clone()

        # generate tokens one at a time
        for _ in range(max_new_tokens):
            last_token = generated[:, -1:]
            logits = self.model(last_token, caches=caches)
            next_token = self.sample_token(logits[:, -1, :], temperature)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
