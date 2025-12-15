"""Simplified MoE implementation for shakespeare training."""

import torch
from torch.profiler import record_function

try:
    import triton_kernels
    import triton_kernels.swiglu
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    from triton_kernels.matmul_ogs import (
        PrecisionConfig,
        FlexCtx,
        FnSpecs,
        FusedActivation,
    )
    from triton_kernels.matmul_ogs import matmul_ogs
    from triton_kernels.numerics import InFlexData
    from triton_kernels.routing import routing
    from triton_kernels.tensor import convert_layout
    from triton_kernels.tensor_details.layout import (
        StridedLayout,
        HopperMXScaleLayout,
        HopperMXValueLayout,
    )
    from triton_kernels.tensor import wrap_torch_tensor, FP4

    HAS_TRITON_KERNELS = True
except ImportError:
    HAS_TRITON_KERNELS = False


def quantize_mx4(w):
    """Quantize weights to MXFP4 format."""
    if HAS_TRITON_KERNELS:
        w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
        w = convert_layout(
            wrap_torch_tensor(w, dtype=FP4), HopperMXValueLayout, mx_axis=1
        )
        w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
        return w, w_scale
    else:
        # fallback: return original weights and dummy scale
        return w, torch.ones(1, device=w.device, dtype=torch.float32)


def swiglu(x, alpha: float = 1.702, limit: float = 7.0, interleaved: bool = True):
    """Swish-Gated Linear Unit activation."""
    if interleaved:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


def moe_simple(
    x,
    wg,
    w1,
    w1_mx,
    w2,
    w2_mx,
    bg,
    b1,
    b2,
    experts_per_token=1,
    num_experts=1,
    swiglu_limit=7.0,
    fused_act=True,
    interleaved=True,
):
    """Simplified MoE for single expert case."""
    if x.numel() == 0:
        return x

    # x might be 2D (batch_size * n_ctx, dim) or 3D (batch_size, n_ctx, dim)
    if x.dim() == 2:
        # Already flattened
        x_flat = x
        batch_size_n_ctx, dim = x.shape
        # We don't know batch_size and n_ctx separately, so use dummy values
        # This should work for the computation
        batch_size = 1
        n_ctx = batch_size_n_ctx
    else:
        batch_size, n_ctx, dim = x.shape
        x_flat = x.view(batch_size * n_ctx, dim)

    # gate (for single expert, just use first expert)
    if num_experts == 1:
        # single expert, no routing needed - use weights directly without indexing
        # w1: (1, hidden_size, intermediate_size * 2) -> (hidden_size, intermediate_size * 2)
        # w2: (1, intermediate_size, hidden_size) -> (intermediate_size, hidden_size)
        w1_use = w1[0, ...]  # (hidden_size, intermediate_size * 2)
        w2_use = w2[0, ...]  # (intermediate_size, hidden_size)
        b1_use = b1[0, ...]  # (intermediate_size * 2,)
        b2_use = b2[0, ...]  # (hidden_size,)
        
        # mlp1: (batch_size * n_ctx, hidden_size) @ (hidden_size, intermediate_size * 2)
        x1 = torch.matmul(x_flat, w1_use) + b1_use
        x1 = swiglu(x1, limit=swiglu_limit, interleaved=interleaved)
        # Ensure x1 has same dtype as w2_use
        x1 = x1.to(w2_use.dtype)
        
        # mlp2: (batch_size * n_ctx, intermediate_size) @ (intermediate_size, hidden_size)
        x2 = torch.matmul(x1, w2_use) + b2_use
        
        output = x2
    else:
        # multi-expert routing
        logits = torch.matmul(x_flat, wg) + bg
        expert_weights, expert_idx = torch.topk(
            torch.softmax(logits, dim=-1), k=experts_per_token, dim=-1, sorted=True
        )
        expert_idx = expert_idx.unsqueeze(-1)
        expert_indices = expert_idx.squeeze(-1)  # (batch_size * n_ctx,)
        
        # mlp1
        w1_selected = w1[expert_indices, ...]  # (batch_size * n_ctx, hidden_size, intermediate_size * 2)
        b1_selected = b1[expert_indices, ...]  # (batch_size * n_ctx, intermediate_size * 2)
        x1 = torch.einsum('bh,bhi->bi', x_flat, w1_selected) + b1_selected
        x1 = swiglu(x1, limit=swiglu_limit, interleaved=interleaved)
        
        # mlp2
        w2_selected = w2[expert_indices, ...]  # (batch_size * n_ctx, intermediate_size, hidden_size)
        b2_selected = b2[expert_indices, ...]  # (batch_size * n_ctx, hidden_size)
        x2 = torch.einsum('bi,bih->bh', x1, w2_selected) + b2_selected
        
        # weighted sum
        output = (x2 * expert_weights).sum(dim=1)
    
    # Reshape back - if input was 2D, return 2D; otherwise return 3D
    if x.dim() == 2:
        return output
    else:
        return output.view(batch_size, n_ctx, dim)


def moe(
    x,
    wg,
    w1,
    w1_mx,
    w2,
    w2_mx,
    bg,
    b1,
    b2,
    experts_per_token=4,
    num_experts=128,
    swiglu_limit=7.0,
    fused_act=True,
    interleaved=True,
):
    """MoE forward pass with triton_kernels if available, fallback otherwise."""
    if x.numel() == 0:
        return x

    if not HAS_TRITON_KERNELS or num_experts == 1:
        return moe_simple(
            x,
            wg,
            w1,
            w1_mx,
            w2,
            w2_mx,
            bg,
            b1,
            b2,
            experts_per_token,
            num_experts,
            swiglu_limit,
            fused_act,
            interleaved,
        )

    # use triton_kernels for multi-expert case
    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=InFlexData()))

    with record_function("wg"):
        logits = matmul_ogs(x, wg, bg, precision_config=pcg)
    with record_function("routing"):
        rdata, gather_indx, scatter_indx = routing(logits, experts_per_token, simulated_ep=1)

    if fused_act:
        assert interleaved, "Fused activation requires interleaved weights"
        with record_function("w1+swiglu"):
            act = FusedActivation(
                FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
                (1.702, swiglu_limit),
                2,
            )
            x = matmul_ogs(
                x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act
            )
    else:
        with record_function("w1"):
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        with record_function("swiglu"):
            x = swiglu(x, limit=swiglu_limit, interleaved=interleaved)

    with record_function("w2"):
        x = matmul_ogs(
            x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2, gammas=rdata.gate_scal
        )
    return x
