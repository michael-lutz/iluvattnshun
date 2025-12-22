"""
Memory-optimized model with both CPU expert offloading AND KV caching.

This eliminates both sources of OOM:
1. CPU offloading: Reduces GPU memory from ~214GB to ~4GB
2. KV caching: Eliminates quadratic activation memory growth
"""
import math
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from .model import (
    ModelConfig, RMSNorm, RotaryEmbedding, swiglu, _apply_rotary_emb
)


@dataclass
class KVCache:
    """Cache for key and value tensors."""
    k: Optional[torch.Tensor] = None  # [batch, seq_len, n_heads, head_dim]
    v: Optional[torch.Tensor] = None  # [batch, seq_len, n_heads, head_dim]


def sdpa_with_cache(
    Q, K, V, S, sm_scale,
    cached_K: Optional[torch.Tensor] = None,
    cached_V: Optional[torch.Tensor] = None,
    sliding_window: int = 0
):
    """Scaled dot-product attention with KV cache support."""
    batch, n_new_tokens, n_heads, q_mult, d_head = Q.shape

    # Concatenate with cache if it exists
    if cached_K is not None:
        K = torch.cat([cached_K, K], dim=1)
        V = torch.cat([cached_V, V], dim=1)

    n_total_tokens = K.shape[1]

    assert K.shape == (batch, n_total_tokens, n_heads, d_head)
    assert V.shape == (batch, n_total_tokens, n_heads, d_head)

    # Expand K and V for GQA
    K = K[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    V = V[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    S = S.reshape(1, n_heads, q_mult, 1, 1).expand(batch, -1, -1, n_new_tokens, -1)

    # Create causal mask for the new tokens
    # New tokens can attend to all previous tokens + themselves causally
    mask = torch.zeros(n_new_tokens, n_total_tokens, device=Q.device, dtype=Q.dtype)
    if n_new_tokens > 1:
        # For prompt processing, apply causal mask
        causal_offset = n_total_tokens - n_new_tokens
        causal_mask = torch.triu(
            Q.new_full((n_new_tokens, n_new_tokens), float('-inf')),
            diagonal=1
        )
        mask[:, causal_offset:] = causal_mask

    # Apply sliding window if enabled
    if sliding_window > 0 and n_total_tokens > sliding_window:
        for i in range(n_new_tokens):
            pos = n_total_tokens - n_new_tokens + i
            window_start = max(0, pos - sliding_window)
            if window_start > 0:
                mask[i, :window_start] = float('-inf')

    # Compute attention
    QK = torch.einsum("bqhmd,bkhmd->bhmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]  # Remove sink token weight
    attn = torch.einsum("bhmqk,bkhmd->bqhmd", W, V)
    return attn.reshape(batch, n_new_tokens, -1), K[:, :, :, 0, :], V[:, :, :, 0, :]


class AttentionBlockCached(torch.nn.Module):
    """Attention with KV caching support."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
        cache_position: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, n_tokens = x.shape[0], x.shape[1]
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, :, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        q = q.view(
            batch,
            n_tokens,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(batch, n_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(batch, n_tokens, self.num_key_value_heads, self.head_dim)

        # Apply RoPE with proper position handling for caching
        # When using cache, we need to apply RoPE at the correct absolute positions
        if cache is not None and cache.k is not None:
            # We're generating, so apply RoPE at position cache_position
            # Create a temp tensor with room for position offset
            seq_len = cache.k.shape[1] + n_tokens
            q_full = torch.zeros(batch, seq_len, *q.shape[2:], device=q.device, dtype=q.dtype)
            k_full = torch.zeros(batch, seq_len, *k.shape[2:], device=k.device, dtype=k.dtype)
            q_full[:, -n_tokens:] = q
            k_full[:, -n_tokens:] = k
            q_full, k_full = self.rope(q_full, k_full)
            q = q_full[:, -n_tokens:]
            k = k_full[:, -n_tokens:]
        else:
            # Processing prompt, apply RoPE normally
            q, k = self.rope(q, k)

        # Use cache if provided
        cached_k = cache.k if cache is not None else None
        cached_v = cache.v if cache is not None else None

        t, new_k, new_v = sdpa_with_cache(
            q, k, v, self.sinks, self.sm_scale,
            cached_k, cached_v, self.sliding_window
        )

        t = self.out(t)
        t = x + t
        return t, new_k, new_v


class MLPBlockOffloaded(torch.nn.Module):
    """MLP block with CPU offloading."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.gpu_device = device if device is not None else torch.device("cuda")
        self.cpu_device = torch.device("cpu")

        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )

        # Keep expert weights on CPU
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2, config.hidden_size),
                device=self.cpu_device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=self.cpu_device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size, config.intermediate_size),
                device=self.cpu_device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=self.cpu_device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_tokens = x.shape[0], x.shape[1]
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        t_flat = t.reshape(batch * n_tokens, -1)
        expert_indices_flat = expert_indices.reshape(batch * n_tokens, -1)
        expert_weights_flat = expert_weights.reshape(batch * n_tokens, -1)

        # Get unique experts and move to GPU
        unique_experts = torch.unique(expert_indices_flat).cpu()

        mlp1_weight_gpu = self.mlp1_weight[unique_experts].to(self.gpu_device, non_blocking=True)
        mlp1_bias_gpu = self.mlp1_bias[unique_experts].to(self.gpu_device, non_blocking=True)
        mlp2_weight_gpu = self.mlp2_weight[unique_experts].to(self.gpu_device, non_blocking=True)
        mlp2_bias_gpu = self.mlp2_bias[unique_experts].to(self.gpu_device, non_blocking=True)

        # Remap indices
        expert_to_idx = {e.item(): i for i, e in enumerate(unique_experts)}
        expert_indices_remapped = torch.tensor(
            [[expert_to_idx[idx.item()] for idx in row] for row in expert_indices_flat],
            device=self.gpu_device
        )

        # MLP forward
        mlp1_weight = mlp1_weight_gpu[expert_indices_remapped, ...]
        mlp1_bias = mlp1_bias_gpu[expert_indices_remapped, ...]
        t_flat = torch.einsum("beck,bk->bec", mlp1_weight, t_flat) + mlp1_bias
        t_flat = swiglu(t_flat, limit=self.swiglu_limit)

        mlp2_weight = mlp2_weight_gpu[expert_indices_remapped, ...]
        mlp2_bias = mlp2_bias_gpu[expert_indices_remapped, ...]
        t_flat = torch.einsum("beck,bek->bec", mlp2_weight, t_flat) + mlp2_bias

        t_flat = torch.einsum("bec,be->bc", t_flat, expert_weights_flat)
        t = t_flat.reshape(batch, n_tokens, -1)

        # Cleanup
        del mlp1_weight_gpu, mlp1_bias_gpu, mlp2_weight_gpu, mlp2_bias_gpu
        del mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias

        return x + t


class TransformerBlockCached(torch.nn.Module):
    """Transformer block with KV caching."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlockCached(config, layer_idx, device)
        self.mlp = MLPBlockOffloaded(config, device)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
        cache_position: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, new_k, new_v = self.attn(x, cache, cache_position)
        x = self.mlp(x)
        return x, new_k, new_v


class TransformerCached(torch.nn.Module):
    """Transformer with KV caching and expert offloading."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlockCached(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: Optional[list[KVCache]] = None,
        cache_position: int = 0
    ) -> Tuple[torch.Tensor, Optional[list[KVCache]]]:
        x = self.embedding(x)

        new_caches = []
        for i, block in enumerate(self.block):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_k, new_v = block(x, cache, cache_position)
            new_caches.append(KVCache(k=new_k, v=new_v))

        x = self.norm(x)
        x = self.unembedding(x)
        return x, new_caches

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "TransformerCached":
        """Load model with KV caching and expert offloading."""
        from .weights import Checkpoint
        import json
        import os

        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = TransformerCached(config=config, device=device)
        model.eval()

        checkpoint = Checkpoint(path, device)

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # Keep MLP expert weights on CPU
            if "mlp" in name and ("mlp1_weight" in name or "mlp2_weight" in name or
                                   "mlp1_bias" in name or "mlp2_bias" in name):
                loaded_tensor = loaded_tensor.cpu()

            try:
                param.data.copy_(loaded_tensor)
            except:
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        print(f"✓ Model loaded with KV caching + expert offloading")
        return model
