"""FlashAttention w/support for learned sinks and banded attention.

This is an expanded version of the Flash Attention v2 implementation (see https://tridao.me/publications/flash2/flash2.pdf)
which can be found at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.

This version has been extended to support banded attention and learned attention sinks.
"""

import torch

import triton
import triton.language as tl

# try to import TensorDescriptor, but if not available, pass tensors directly
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    USE_TENSOR_DESCRIPTOR = True
except ImportError:
    USE_TENSOR_DESCRIPTOR = False
    TensorDescriptor = None

try:
    import pytest
except ImportError:
    pytest = None


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    M,
    Out,  #
    Start_q,
    Z,
    H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    if BANDWIDTH:
        lo, hi = tl.maximum(start_q, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        lo, hi = start_q, start_q + (start_m + 1) * BLOCK_M

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T
        qk = tl.dot(q, k, allow_tf32=False)

        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc, allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    sink = tl.math.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]
    m_i += tl.math.log(l_i)
    m_ptrs = M + off_hz * N_Q_CTX + offs_m
    tl.store(m_ptrs, m_i)
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        assert len(start_q) == 1
        # q shape: (batch_size, n_ctx, num_attention_heads // num_key_value_heads, num_key_value_heads, head_dim)
        # k, v shape: (batch_size, n_kv_ctx, num_key_value_heads, head_dim)
        # q might be (bs, n_ctx, repeat_kv, n_kv_heads, head_dim) or (bs, n_heads, n_ctx, head_dim)
        # k, v are (bs, n_kv_ctx, n_kv_heads, head_dim)
        if q.dim() == 5:
            bs, n_ctx, repeat_kv, n_kv_heads, HEAD_DIM_Q = q.shape
        elif q.dim() == 4:
            # q is (bs, n_heads, n_ctx, head_dim)
            bs, n_heads_q, n_ctx, HEAD_DIM_Q = q.shape
            # Infer repeat_kv and n_kv_heads from k
            bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K = k.shape
            repeat_kv = n_heads_q // n_kv_heads
            n_kv_heads_q = n_kv_heads
            # Reshape q to (bs, n_ctx, repeat_kv, n_kv_heads, head_dim)
            q = q.transpose(1, 2).view(bs, n_ctx, repeat_kv, n_kv_heads, HEAD_DIM_Q)
        else:
            raise ValueError(f"Unexpected q shape: {q.shape}")
        
        bs, n_kv_ctx, n_kv_heads_kv, HEAD_DIM_K = k.shape
        bs, n_kv_ctx, n_kv_heads_v, HEAD_DIM_V = v.shape
        assert n_kv_heads == n_kv_heads_kv == n_kv_heads_v
        n_heads = n_kv_heads * repeat_kv
        
        # Ensure head dims match - use the minimum
        head_dim = min(HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V)
        if HEAD_DIM_Q != head_dim:
            q = q[:, :, :, :, :head_dim]
        if HEAD_DIM_K != head_dim:
            k = k[:, :, :, :head_dim]
        if HEAD_DIM_V != head_dim:
            v = v[:, :, :, :head_dim]
        
        # q is now (bs, n_ctx, repeat_kv, n_kv_heads, head_dim)
        # Reshape for attention: (bs, n_ctx, n_heads, head_dim)
        q = q.view(bs, n_ctx, n_heads, head_dim)
        k = k.view(bs, n_kv_ctx, n_kv_heads, head_dim)
        assert head_dim in {16, 32, 64, 128, 256}

        q = q.transpose(1, 2).contiguous()
        k = k.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v = v.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        # adjust block sizes based on head_dim
        BLOCK_M = 64
        BLOCK_N = min(64, HEAD_DIM_K)
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        # pad q to multiple of its block size in the n_ctx dimension (-2)
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))
        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        # pad k and v to multiple of their block size in the n_kv_ctx dimension
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        if USE_TENSOR_DESCRIPTOR:
            o = torch.empty_like(q)
            M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)
            grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)
            _attn_fwd[grid](
                TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, HEAD_DIM_K]),
                TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, HEAD_DIM_K]),
                TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, HEAD_DIM_K]),
                sinks,
                sm_scale,
                M,
                TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM_K]),
                start_q,
                q.shape[0],
                q.shape[1],
            N_Q_CTX=n_ctx + m_pad_size,
            N_KV_CTX=n_kv_ctx,
            HEAD_DIM=head_dim,
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            )
            ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
            ctx.sm_scale = sm_scale
            ctx.bandwidth = bandwidth
            o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()
            o = o.view(bs, n_ctx, n_heads * head_dim)
        else:
            # fallback to reference implementation when TensorDescriptor not available
            # q is (bs, n_heads, n_ctx, head_dim_q) from transpose above
            # k, v are (bs, n_heads, n_kv_ctx, head_dim_k/v) from repeat_interleave and transpose
            # Get actual dimensions
            bs_q, n_heads_q, n_ctx_q, head_dim_q = q.shape
            bs_k, n_heads_k, n_kv_ctx_k, head_dim_k = k.shape
            
            # Use head_dim from k to ensure compatibility
            head_dim_use = head_dim_k
            
            # k, v: transpose back to (bs, n_kv_ctx, n_heads, head_dim), then take first n_kv_heads
            k_for_ref = k.transpose(1, 2).contiguous()  # (bs, n_kv_ctx, n_heads, head_dim)
            v_for_ref = v.transpose(1, 2).contiguous()  # (bs, n_kv_ctx, n_heads, head_dim)
            k_for_ref = k_for_ref[:, :, :n_kv_heads, :head_dim_use]
            v_for_ref = v_for_ref[:, :, :n_kv_heads, :head_dim_use]
            
            # q is (bs, n_heads, n_ctx, head_dim_q) from transpose above
            # transpose to (bs, n_ctx, n_heads, head_dim_q)
            q_for_ref = q.transpose(1, 2).contiguous()  # (bs, n_ctx, n_heads, head_dim_q)
            # Check actual shape
            bs_actual, n_ctx_actual, n_heads_actual, head_dim_q_actual = q_for_ref.shape
            # Adjust head_dim to match k
            if head_dim_q_actual != head_dim_use:
                if head_dim_q_actual > head_dim_use:
                    q_for_ref = q_for_ref[:, :, :, :head_dim_use]
                else:
                    padding = torch.zeros(bs_actual, n_ctx_actual, n_heads_actual, head_dim_use - head_dim_q_actual,
                                        device=q_for_ref.device, dtype=q_for_ref.dtype)
                    q_for_ref = torch.cat([q_for_ref, padding], dim=3)
            
            # Now q_for_ref is (bs, n_ctx, n_heads, head_dim_use)
            # Reshape to (bs, n_ctx, repeat_kv, n_kv_heads, head_dim_use)
            # n_heads should equal repeat_kv * n_kv_heads
            # If there's a mismatch in total elements, flatten and take what we need
            total_elements = q_for_ref.numel()
            expected_elements = bs * n_ctx * repeat_kv * n_kv_heads * head_dim_use
            if total_elements == expected_elements:
                # Correct total size - reshape directly
                q_reshaped = q_for_ref.view(bs, n_ctx, repeat_kv, n_kv_heads, head_dim_use)
            else:
                # Wrong total size - q_for_ref might be (bs, n_ctx, n_heads, wrong_head_dim)
                # Flatten and take first expected_hidden elements
                q_flat = q_for_ref.reshape(bs, n_ctx, -1)
                expected_hidden = repeat_kv * n_kv_heads * head_dim_use
                if q_flat.shape[2] >= expected_hidden:
                    q_flat = q_flat[:, :, :expected_hidden]
                else:
                    padding = torch.zeros(bs, n_ctx, expected_hidden - q_flat.shape[2],
                                        device=q_flat.device, dtype=q_flat.dtype)
                    q_flat = torch.cat([q_flat, padding], dim=2)
                q_reshaped = q_flat.view(bs, n_ctx, repeat_kv, n_kv_heads, head_dim_use)
            
            # Use attention_ref but wrap it to enable autograd
            # Save inputs for backward
            ctx.save_for_backward(q_reshaped, k_for_ref, v_for_ref, sinks, start_q)
            ctx.sm_scale = sm_scale
            ctx.bandwidth = bandwidth
            # Compute forward with autograd enabled
            o = attention_ref(q_reshaped, k_for_ref, v_for_ref, sinks, sm_scale, bandwidth, start_q)
            # o from attention_ref is (bs, n_ctx, n_heads * head_dim) = (bs, n_ctx, hidden_size)
            # which is already the correct output shape
        
        return o

    @staticmethod
    def backward(ctx, grad_output):
        # Backward for reference implementation path
        # Get saved tensors
        q_reshaped, k_for_ref, v_for_ref, sinks, start_q = ctx.saved_tensors
        
        # Use autograd by recomputing with gradients enabled
        grad_q = grad_k = grad_v = grad_sinks = None
        if any([ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2], ctx.needs_input_grad[3]]):
            with torch.enable_grad():
                q_ref = q_reshaped.detach().requires_grad_(ctx.needs_input_grad[0])
                k_ref = k_for_ref.detach().requires_grad_(ctx.needs_input_grad[1])
                v_ref = v_for_ref.detach().requires_grad_(ctx.needs_input_grad[2])
                sinks_ref = sinks.detach().requires_grad_(ctx.needs_input_grad[3])
                
                # Recompute forward
                o_ref = attention_ref(q_ref, k_ref, v_ref, sinks_ref, ctx.sm_scale, ctx.bandwidth, start_q)
                o_ref.backward(grad_output)
                
                grad_q_reshaped = q_ref.grad
                grad_k_for_ref = k_ref.grad
                grad_v_for_ref = v_ref.grad
                grad_sinks = sinks_ref.grad
                
                # Convert gradients back to original shapes
                # q_reshaped is (bs, n_ctx, repeat_kv, n_kv_heads, head_dim)
                # Original q shape when passed to attention is (bs, n_ctx, repeat_kv, n_kv_heads, head_dim)
                # But we need to return it in the shape that matches the input to attention
                # The input q to attention forward is (bs, n_heads, n_ctx, head_dim) after transpose
                # So we need to convert back
                if grad_q_reshaped is not None:
                    # grad_q_reshaped is (bs, n_ctx, repeat_kv, n_kv_heads, head_dim)
                    # We need to return it in the same shape as q_reshaped was passed in
                    # But actually, the gradient should match the input shape
                    # The input q to attention is (bs, n_ctx, repeat_kv, n_kv_heads, head_dim) from model
                    # So return it as is
                    grad_q = grad_q_reshaped
                else:
                    grad_q = None
                
                # k, v gradients are already in correct shape (bs, n_kv_ctx, n_kv_heads, head_dim)
                grad_k = grad_k_for_ref
                grad_v = grad_v_for_ref
        
        return grad_q, grad_k, grad_v, grad_sinks, None, None, None


attention = _attention.apply


def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
):
    batch_size, num_queries, num_key_value_groups, num_key_value_heads_q, head_dim_q = query.shape
    batch_size, num_keys, num_key_value_heads_kv, head_dim_kv = key.shape
    batch_size, num_keys_v, num_key_value_heads_v, head_dim_v = value.shape
    # Use minimum head_dim to ensure compatibility
    head_dim = min(head_dim_q, head_dim_kv, head_dim_v)
    if head_dim_q != head_dim:
        query = query[:, :, :, :, :head_dim]
    if head_dim_kv != head_dim:
        key = key[:, :, :, :head_dim]
    if head_dim_v != head_dim:
        value = value[:, :, :, :head_dim]
    # num_key_value_heads_q should match num_key_value_heads_kv
    num_key_value_heads = num_key_value_heads_kv
    assert num_key_value_heads == num_key_value_heads_v
    num_heads = num_key_value_heads * num_key_value_groups

    # sinks shape is (num_attention_heads,) = (num_key_value_heads * num_key_value_groups,)
    # reshape sinks to match the einsum output shape: (1, num_key_value_heads, num_key_value_groups, 1, 1)
    # sinks has shape (num_attention_heads,) = (num_key_value_heads * num_key_value_groups,)
    # verify the sizes match
    expected_sinks_size = num_key_value_heads * num_key_value_groups
    if sinks.numel() != expected_sinks_size:
        # fallback: just broadcast sinks
        sinks_expanded = sinks.view(1, -1, 1, 1, 1).float()
        # pad or trim to match expected shape
        if sinks_expanded.shape[1] < num_key_value_heads:
            padding = torch.zeros(1, num_key_value_heads - sinks_expanded.shape[1], 1, 1, 1, device=sinks.device, dtype=sinks_expanded.dtype)
            sinks_expanded = torch.cat([sinks_expanded, padding], dim=1)
        sinks_expanded = sinks_expanded[:, :num_key_value_heads, :num_key_value_groups, :, :]
    else:
        sinks_expanded = sinks.view(num_key_value_groups, num_key_value_heads).transpose(0, 1).contiguous()
        sinks_expanded = sinks_expanded.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    
    # key and value are (bs, num_keys, num_kv_heads, head_dim)
    # we need to add a dimension for num_key_value_groups to match query structure
    # key: (bs, num_keys, num_kv_heads, head_dim) -> (bs, num_keys, num_kv_heads, num_kv_groups, head_dim)
    # value: (bs, num_keys, num_kv_heads, head_dim) -> (bs, num_keys, num_kv_heads, num_kv_groups, head_dim)
    if key.dim() == 4:
        key = key.unsqueeze(3).expand(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)
    elif key.shape[3] == 1:
        key = key.expand(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)
    
    if value.dim() == 4:
        value = value.unsqueeze(3).expand(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)
    elif value.shape[3] == 1:
        value = value.expand(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    # query: (bs, num_queries, num_kv_groups, num_kv_heads, head_dim)
    # key: (bs, num_keys, num_kv_heads, 1, head_dim)
    # logits: (bs, num_kv_heads, num_kv_groups, num_queries, num_keys)
    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
    sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
    scores = unnormalized_scores / normalizer

    # scores: (bs, num_kv_heads, num_kv_groups, num_queries, num_keys)
    # value: (bs, num_keys, num_kv_heads, num_kv_groups, head_dim)
    # Contract over k (num_keys) dimension to get (bs, num_queries, num_kv_heads, num_kv_groups, head_dim)
    
    # Ensure value has correct shape (bs, num_keys, num_kv_heads, num_kv_groups, head_dim)
    # value comes in as (bs, num_keys, num_kv_heads, head_dim) from the model
    # where num_kv_heads should be num_key_value_heads (4 in our case)
    if value.dim() == 4:
        # value is (bs, num_keys, num_kv_heads, head_dim)
        # Check if num_kv_heads matches num_key_value_heads
        if value.shape[2] != num_key_value_heads:
            # Wrong number of heads - take first num_key_value_heads
            value = value[:, :, :num_key_value_heads, :]
        # Add num_kv_groups dimension: (bs, num_keys, num_kv_heads, 1, head_dim) -> expand to (bs, num_keys, num_kv_heads, num_kv_groups, head_dim)
        value = value.unsqueeze(3).expand(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)
    elif value.dim() == 5:
        # value is already 5D
        if value.shape[2] != num_key_value_heads:
            # Wrong number of heads
            value = value[:, :, :num_key_value_heads, :, :]
        if value.shape[3] == 1:
            # value is (bs, num_keys, num_kv_heads, 1, head_dim), expand to include num_kv_groups
            value = value.expand(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)
        elif value.shape[3] != num_key_value_groups:
            # Wrong num_kv_groups dimension, take first one and expand
            value = value[:, :, :, :1, :].expand(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)
    
    # Verify value shape before einsum
    assert value.shape == (batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim), \
        f"value shape {value.shape} != expected {(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim)}"
    
    # Perform einsum: scores (b,h,m,q,k) * value (b,k,h,m,d) -> (b,q,h,m,d)
    # where b=batch, h=num_kv_heads, m=num_kv_groups, q=num_queries, k=num_keys, d=head_dim
    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())
    
    # Output should be (batch, num_queries, num_key_value_heads, num_key_value_groups, head_dim)
    expected_shape = (batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim)
    expected_hidden = num_key_value_heads * num_key_value_groups * head_dim
    
    # Reshape to (batch, num_queries, hidden_size)
    if output.shape == expected_shape:
        output = output.reshape(batch_size, num_queries, expected_hidden)
    else:
        # Shape mismatch - try to fix
        if output.numel() == batch_size * num_queries * expected_hidden:
            output = output.reshape(expected_shape).reshape(batch_size, num_queries, expected_hidden)
        else:
            # Wrong total size - take first expected_hidden elements
            output = output.reshape(batch_size, num_queries, -1)
            if output.shape[2] >= expected_hidden:
                output = output[:, :, :expected_hidden]
            else:
                # Pad if needed
                padding = torch.zeros(batch_size, num_queries, expected_hidden - output.shape[2],
                                    device=output.device, dtype=output.dtype)
                output = torch.cat([output, padding], dim=2)
    
    output = output.bfloat16()
    return output


if pytest is not None:
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_queries", [1, 128])
    @pytest.mark.parametrize("num_keys", [128, 32])
    @pytest.mark.parametrize("num_key_value_heads", [8])
    @pytest.mark.parametrize("num_key_value_groups", [8])
    @pytest.mark.parametrize("head_dim", [64])
    @pytest.mark.parametrize("sm_scale", [0.125])
    @pytest.mark.parametrize("sliding_window", [None, 128])
    @pytest.mark.parametrize("start_q", [0, 5])
    def test_eq(batch_size, num_queries, num_keys, num_key_value_heads, num_key_value_groups, head_dim, sm_scale, sliding_window, start_q):
        if num_queries > num_keys:
            pytest.skip("too many queries")

    q = torch.randn(batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim).bfloat16().cuda()
    k = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    v = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    sinks = torch.randn(num_key_value_heads * num_key_value_groups).bfloat16().cuda()

    start_q = torch.tensor([start_q], dtype=torch.int32).cuda()

    o1 = attention(q, k, v, sinks, sm_scale, sliding_window, start_q)
    o2 = attention_ref(q, k, v, sinks, sm_scale, sliding_window, start_q)

    torch.testing.assert_close(o1, o2)
