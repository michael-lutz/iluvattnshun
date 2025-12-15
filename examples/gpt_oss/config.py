"""Simplified model config for shakespeare training."""

import math
from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    """Model configuration for triton transformer."""

    num_hidden_layers: int = 8
    num_experts: int = 1
    experts_per_token: int = 1
    vocab_size: int = 65
    hidden_size: int = 128
    intermediate_size: int = 512
    swiglu_limit: float = 7.0
    head_dim: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    sliding_window: int = 0
    initial_context_length: int = 128
    rope_theta: float = 1024.0
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class RMSNorm(torch.nn.Module):
    """RMS normalization layer."""

    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        result = (t * self.scale).to(dtype)
        # ensure output matches input dtype
        return result
