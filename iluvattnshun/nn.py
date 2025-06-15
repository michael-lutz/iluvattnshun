"""Includes basic standard neural network modules."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """A simple attention module with KV caching support."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
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

        self.dropout = nn.Dropout(dropout)  # TODO: investigate dropout effect...
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn_weights: bool = False,
        return_new_kv_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Forward pass through the attention layer.
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Optional mask tensor of shape (batch_size, seq_len)
            is_causal: Whether to use causal masking
            kv_cache: Optional tuple of (cached_keys, cached_values)
            return_attn_weights: Whether to return attention weights
            return_new_kv_cache: Whether to return new KV cache

        Returns:
            output: Output tensor of shape (batch_size, seq_len, embed_dim)
            attn_weights: Optional attention weights
            new_kv_cache: New KV cache tuple
        """
        # project queries, keys, and values
        # TODO: eventually allow for different projections for q, k, v
        q = self.q_proj(query)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(key)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(value)  # (batch_size, seq_len, embed_dim)

        # reshape for multi-head attention
        batch_size = query.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # handle kv cache if provided
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # concatenate along sequence dimension
            v = torch.cat([cached_v, v], dim=2)

        # compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # apply causal mask if needed
        if is_causal:
            mask = torch.triu(torch.ones(attn.size(-2), attn.size(-1), device=attn.device), diagonal=1)
            attn = attn.masked_fill(mask.bool(), float("-inf"))

        # apply key padding mask if provided
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # apply softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)

        attention_weights = attn_weights if return_attn_weights else None
        new_kv_cache = (k, v) if return_new_kv_cache else None
        return output, attention_weights, new_kv_cache


class TransformerLayer(nn.Module):
    """A single transformer layer."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = Attention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_attn_weights: bool = False,
        return_new_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass through the transformer layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            kv_cache: Optional tuple of (key, value) tensors from previous forward pass

        Returns:
            Tuple of (output, new_kv_cache)
        """
        x_norm = self.norm1(x)  # Pre-LN Xiong et al., 2020 (https://arxiv.org/abs/2002.04745v1)

        if kv_cache is not None:
            k, v = kv_cache
            attn_out, attn_weights, new_kv_cache = self.attention(
                x_norm,
                k,
                v,
                is_causal=True,
                return_attn_weights=return_attn_weights,
                return_new_kv_cache=return_new_kv_cache,
            )
        else:
            attn_out, attn_weights, new_kv_cache = self.attention(
                x_norm,
                x_norm,
                x_norm,
                is_causal=True,
                return_attn_weights=return_attn_weights,
                return_new_kv_cache=return_new_kv_cache,
            )

        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x, attn_weights, new_kv_cache


class MultilayerTransformer(nn.Module):
    """A multilayer transformer model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, scale_grad_by_freq=True)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model**-0.5)  # Scale during initialization
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_attn_weights: bool = False,
        return_new_kv_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
        list[torch.Tensor] | None,
    ]:
        """Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            kv_cache: Optional list of (key, value) tuples for each layer
            return_attn_weights: Whether to return attention weights
            return_new_kv_cache: Whether to return new KV cache

        Returns:
            Tuple of (output logits, new kv_cache)
        """
        batch_size, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.pos_embedding(pos)

        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if return_new_kv_cache else None
        attn_weights: list[torch.Tensor] | None = [] if return_attn_weights else None
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_attn_weights, layer_kv_cache = layer(
                x,
                layer_kv_cache,
                return_attn_weights=return_attn_weights,
                return_new_kv_cache=return_new_kv_cache,
            )

            if return_new_kv_cache and new_kv_cache is not None and layer_kv_cache is not None:
                new_kv_cache.append(layer_kv_cache)
            if return_attn_weights and attn_weights is not None and layer_attn_weights is not None:
                attn_weights.append(layer_attn_weights)

        logits: torch.Tensor = self.output(x)
        return logits, new_kv_cache, attn_weights

    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            prompt: Initial prompt tensor of shape (batch_size, prompt_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated sequence including prompt
        """
        generated = prompt.clone()
        kv_cache = None

        for _ in range(max_new_tokens):
            # get logits using most recent tokens
            logits, attn_weights, kv_cache = self(
                generated, kv_cache, return_attn_weights=False, return_new_kv_cache=True
            )
            next_token_logits = logits[:, -1, :] / temperature

            # sample and add to generated sequence
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
