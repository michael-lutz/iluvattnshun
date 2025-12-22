"""
Memory-optimized model with CPU offloading for MoE experts.

This version keeps expert weights on CPU and moves them to GPU on-demand,
dramatically reducing GPU memory requirements from ~214GB to ~20GB.
"""
import torch
from .model import (
    ModelConfig, RMSNorm, AttentionBlock, swiglu,
    Transformer as BaseTransformer, TransformerBlock as BaseTransformerBlock
)


class MLPBlockOffloaded(torch.nn.Module):
    """MLP block with CPU offloading - keeps experts on CPU, moves to GPU on demand."""

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

        # Keep expert weights on CPU to save GPU memory
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2,
                    config.hidden_size,
                ),
                device=self.cpu_device,  # CPU!
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=self.cpu_device,  # CPU!
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size,
                ),
                device=self.cpu_device,  # CPU!
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=self.cpu_device,  # CPU!
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

        # Reshape for expert processing
        t_flat = t.reshape(batch * n_tokens, -1)
        expert_indices_flat = expert_indices.reshape(batch * n_tokens, -1)
        expert_weights_flat = expert_weights.reshape(batch * n_tokens, -1)

        # Get unique experts used in this batch to minimize transfers
        unique_experts = torch.unique(expert_indices_flat).cpu()  # Need CPU indices for CPU tensor

        # Move only required experts to GPU
        mlp1_weight_gpu = self.mlp1_weight[unique_experts].to(self.gpu_device, non_blocking=True)
        mlp1_bias_gpu = self.mlp1_bias[unique_experts].to(self.gpu_device, non_blocking=True)
        mlp2_weight_gpu = self.mlp2_weight[unique_experts].to(self.gpu_device, non_blocking=True)
        mlp2_bias_gpu = self.mlp2_bias[unique_experts].to(self.gpu_device, non_blocking=True)

        # Remap expert indices to the subset we loaded
        expert_to_idx = {e.item(): i for i, e in enumerate(unique_experts)}
        expert_indices_remapped = torch.tensor(
            [[expert_to_idx[idx.item()] for idx in row] for row in expert_indices_flat],
            device=self.gpu_device
        )

        # MLP #1
        mlp1_weight = mlp1_weight_gpu[expert_indices_remapped, ...]
        mlp1_bias = mlp1_bias_gpu[expert_indices_remapped, ...]
        t_flat = torch.einsum("beck,bk->bec", mlp1_weight, t_flat) + mlp1_bias
        t_flat = swiglu(t_flat, limit=self.swiglu_limit)

        # MLP #2
        mlp2_weight = mlp2_weight_gpu[expert_indices_remapped, ...]
        mlp2_bias = mlp2_bias_gpu[expert_indices_remapped, ...]
        t_flat = torch.einsum("beck,bek->bec", mlp2_weight, t_flat) + mlp2_bias

        # Weighted sum of experts
        t_flat = torch.einsum("bec,be->bc", t_flat, expert_weights_flat)

        # Reshape back
        t = t_flat.reshape(batch, n_tokens, -1)

        # Clean up GPU expert weights immediately to free memory
        del mlp1_weight_gpu, mlp1_bias_gpu, mlp2_weight_gpu, mlp2_bias_gpu
        del mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias

        return x + t


class TransformerBlockOffloaded(torch.nn.Module):
    """Transformer block using offloaded MLP."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlockOffloaded(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class TransformerOffloaded(BaseTransformer):
    """Transformer with CPU-offloaded experts."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        # Call nn.Module.__init__ directly to skip BaseTransformer.__init__
        torch.nn.Module.__init__(self)

        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlockOffloaded(config, layer_idx, device)
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

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "TransformerOffloaded":
        """Load model with expert offloading enabled."""
        from .weights import Checkpoint
        import json
        import os

        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = TransformerOffloaded(config=config, device=device)
        model.eval()

        # Load weights - experts will be on CPU
        checkpoint = Checkpoint(path, device)

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # Keep MLP expert weights on CPU
            if "mlp" in name and ("mlp1_weight" in name or "mlp2_weight" in name or
                                   "mlp1_bias" in name or "mlp2_bias" in name):
                # Move to CPU instead of GPU
                loaded_tensor = loaded_tensor.cpu()

            try:
                param.data.copy_(loaded_tensor)
            except:
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        print(f"✓ Model loaded with expert offloading (experts on CPU)")
        return model
