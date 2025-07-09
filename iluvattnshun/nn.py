"""Includes basic standard neural network modules."""

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) implementation.

    Implemented as module to avoid recomputing the inv_freq tensor.
    """

    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        # getting the inverse frequency θ_i in the RoPE paper
        inv_freq = base ** (-torch.arange(0, d_model, 2).float() / d_model)  # (d_model // 2,)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Forward pass through the rotary positional embedding.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            offset: Offset to apply to the sequence length

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # TODO: optimize training by caching sin and cos values

        d_model = x.size(-1)
        seq_len = x.size(-2)
        assert d_model % 2 == 0, "d_model must be even"

        # getting positional embeddings pre-rotation
        pos = torch.arange(start=offset, end=seq_len + offset, device=x.device, dtype=torch.float32)
        pos_emb = torch.einsum("i,j->ij", pos, self.inv_freq)  # (seq_len, d_model // 2)

        # getting the sin and cos values
        sin = torch.sin(pos_emb)
        cos = torch.cos(pos_emb)

        # applying the rotary positional embedding (keeping interleaved)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos
        assert x_rotated.shape == x.shape, "something went horribly wrong."  # TODO: remove if never hit...
        return x_rotated


class Attention(nn.Module):
    """A simple attention module with KV caching support."""

    def __init__(self, embed_dim: int, num_heads: int, rope_base: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # RoPE
        self.rotary_embedding = RotaryEmbedding(self.head_dim, rope_base)

        self.scale = self.head_dim**-0.5

        # Optional override functions for patching
        self.qkv_override: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None  # override q, k, v (batch, heads, seq, head_dim) before attn
        self.attn_weights_override: Callable[[torch.Tensor], torch.Tensor] | None = (
            None  # override attn map (batch, heads, seq, seq) pos softmax
        )

    def forward(
        self,
        x: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        return_attn_weights: bool = False,
        return_new_kv_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        """Forward pass through the attention layer.
        Args:
            x: Query tensor of shape (batch, seq, embed)
            key: Key tensor of shape (batch, num_heads, seq, d_model/num_heads)
            value: Value tensor of shape (batch, num_heads, seq, d_model/num_heads)
            key_padding_mask: Optional mask tensor of shape (batch, seq)
            is_causal: Whether to use causal masking
            return_attn_weights: Whether to return attention weights
            return_new_kv_cache: Whether to return new KV cache

        Returns:
            output: Output tensor of shape (batch_size, seq_len, embed_dim)
            new_kv_cache: New KV cache tuple
            attn_weights: Optional attention weights
        """
        # project queries, keys, and values
        # TODO: eventually allow for different projections for q, k, v
        q = self.q_proj(x)  # (batch, seq, embed)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for multi-head attention
        batch_size = x.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # handle kv cache if provided (will already have RoPE applied)
        if key is not None and value is not None:
            assert (
                len(key.shape) == 4
                and key.shape[0] == batch_size
                and key.shape[1] == self.num_heads
                and value.shape == key.shape
            ), "key and value must have shape (batch, heads, seq, head_dim)"

            # apply RoPE with offset
            num_cached_tokens = key.size(2)
            q = self.rotary_embedding(q, offset=num_cached_tokens)
            k = self.rotary_embedding(k, offset=num_cached_tokens)

            k = torch.cat([key, k], dim=2)  # (batch, heads, seq + cached, head_dim)
            v = torch.cat([value, v], dim=2)
        else:
            # apply RoPE without offset
            q = self.rotary_embedding(q)
            k = self.rotary_embedding(k)

        # apply QKV override if provided
        if self.qkv_override is not None:
            q, k, v = self.qkv_override(q, k, v)

        # compute attention scores
        assert x.device == k.device == v.device, "x, k, v must be on the same device"
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # apply causal mask if no KV cache
        if is_causal and key is None and value is None:
            mask = torch.triu(torch.ones(attn.size(-2), attn.size(-1), device=attn.device), diagonal=1)
            attn = attn.masked_fill(mask.bool(), float("-inf"))

        # apply key padding mask if provided
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # apply softmax and compute output
        attn_weights = F.softmax(attn, dim=-1)

        # apply attention weights override if provided
        if self.attn_weights_override is not None:
            attn_weights = self.attn_weights_override(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)

        returned_new_kv_cache = (k, v) if return_new_kv_cache else None
        returned_attn_weights = attn_weights if return_attn_weights else None
        return output, returned_new_kv_cache, returned_attn_weights


class TransformerLayer(nn.Module):
    """A single transformer layer."""

    def __init__(
        self, d_model: int, n_heads: int, rope_base: float, dropout_attn: float = 0.1, dropout_mlp: float = 0.1
    ):
        super().__init__()
        self.attention = Attention(d_model, n_heads, rope_base)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_attn = Dropout(dropout_attn)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.dropout_mlp = Dropout(dropout_mlp)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_new_kv_cache: bool = False,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        """Forward pass through the transformer layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            kv_cache: Optional tuple of (key, value) tensors from previous forward pass

        Returns:
            Tuple of (output, new_kv_cache, attn_weights)
        """
        x_norm = self.norm1(x)  # Pre-LN Xiong et al., 2020 (https://arxiv.org/abs/2002.04745v1)

        if kv_cache is not None:
            k, v = kv_cache
            attn_out, new_kv_cache, attn_weights = self.attention(
                x_norm,
                key=k,
                value=v,
                is_causal=True,
                return_attn_weights=return_attn_weights,
                return_new_kv_cache=return_new_kv_cache,
            )
        else:
            attn_out, new_kv_cache, attn_weights = self.attention(
                x_norm,
                is_causal=True,
                return_attn_weights=return_attn_weights,
                return_new_kv_cache=return_new_kv_cache,
            )

        x = x + self.dropout_attn(attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout_mlp(mlp_out)

        return x, new_kv_cache, attn_weights


class MultilayerTransformer(nn.Module):
    """A multilayer transformer model with RoPE."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        rope_base: float = 10000.0,
        dropout_attn: float = 0.1,
        dropout_mlp: float = 0.1,
        dropout_emb: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, scale_grad_by_freq=True)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model**-0.5)
        self.dropout_emb = Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model, n_heads, rope_base=rope_base, dropout_attn=dropout_attn, dropout_mlp=dropout_mlp
                )
                for _ in range(n_layers)
            ]
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_new_kv_cache: bool = False,
        return_attn_weights: bool = False,
        return_xs: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        """Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            kv_cache: Optional list of (key, value) tuples for each layer
            return_new_kv_cache: Whether to return new KV cache
            return_attn_weights: Whether to return attention weights
            return_xs: Whether to return intermediate x, starting at token emb

        Returns:
            Tuple of (output logits, new kv_cache, attention weights)
        """
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.dropout_emb(x)
        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if return_new_kv_cache else None
        attn_weights: list[torch.Tensor] | None = [] if return_attn_weights else None
        xs: list[torch.Tensor] | None = [x.clone()] if return_xs else None

        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_kv_cache, layer_attn_weights = layer(
                x,
                layer_kv_cache,
                return_new_kv_cache=return_new_kv_cache,
                return_attn_weights=return_attn_weights,
            )

            if return_new_kv_cache and new_kv_cache is not None and layer_kv_cache is not None:
                new_kv_cache.append(layer_kv_cache)
            if return_attn_weights and attn_weights is not None and layer_attn_weights is not None:
                attn_weights.append(layer_attn_weights)
            if return_xs and xs is not None:
                xs.append(x.clone())

        logits: torch.Tensor = self.output(x)
        return logits, new_kv_cache, attn_weights, xs

    def sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
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

        # run once on the whole prompt to prime the cache
        _, kv_cache, _ = self(prompt, return_attn_weights=False, return_new_kv_cache=True)
        generated = prompt  # start with the full prompt

        for _ in range(max_new_tokens):
            # keep feeding only the last token and append to the full sequence
            last_token = generated[:, -1:].clone()
            logits, kv_cache, _ = self(last_token, kv_cache, return_attn_weights=False, return_new_kv_cache=True)
            next_token = self.sample_token(logits[:, -1, :], temperature)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
