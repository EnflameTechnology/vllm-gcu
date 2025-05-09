from typing import List, Optional, Tuple

import tops_extension.torch  # noqa: F401
import torch
import torch_gcu  # noqa: F401

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

from vllm.platforms import current_platform
import vllm_gcu._C  # noqa: F401


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale_float: float,
    v_scale_float: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
    k_zero_float: float = 0.0,
    v_zero_float: float = 0.0,
    out_scales: Optional[torch.Tensor] = None,
) -> None:
    # TODO change hard code
    torch.ops._C.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale_float,
        v_scale_float,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
        k_zero_float,
        v_zero_float,
        out_scales,
    )


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale_float: float,
    v_scale_float: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
    k_zero_float: float = 0.0,
    v_zero_float: float = 0.0,
    out_scales: Optional[torch.Tensor] = None,
) -> None:
    # TODO change hard code
    torch.ops._C.paged_attention_v2(
        out,
        exp_sum,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale_float,
        v_scale_float,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
        k_zero_float,
        v_zero_float,
        out_scales,
    )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale_float: float,
    v_scale_float: float,
    k_zero_float: float = 0.0,
    v_zero_float: float = 0.0,
) -> None:
    # TODO change hard code
    torch.ops._C_cache_ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale_float,
        v_scale_float,
        k_zero_float,
        v_zero_float,
    )


def copy_blocks(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    block_mapping: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(
    src: torch.Tensor, dst: torch.Tensor, block_mapping: torch.Tensor
) -> None:
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_mapping)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    if query.numel() == 0:
        return

    torch.ops._C.rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )


def batched_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: torch.Tensor,
) -> None:
    raise NotImplementedError


# layer norm ops
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def advance_step(
    num_seqs: int,
    num_queries: int,
    block_size: int,
    input_tokens: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    input_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    if current_platform.get_device_capability().to_int() != 130:
        raise NotImplementedError

    return torch.ops._C.advance_step_xformers(
        num_seqs,
        num_queries,
        block_size,
        input_tokens,
        sampled_token_ids,
        input_positions,
        seq_lens,
        slot_mapping,
        block_tables,
    )


# quantization ops
# awq
def awq_gemm_gcu(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    split_k_iters: int,
    bias=None,
    group_size=128,
) -> torch.Tensor:
    return torch.ops._C.awq_gemm_gcu(
        input, qweight, qzeros, scales, split_k_iters, bias, group_size
    )


@register_fake("_C::awq_gemm_gcu")
def _awq_gemm_gcu_fake(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    split_k_iters: int,
    bias=None,
    group_size=128,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (qweight.shape[-1],)
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


# gptq
def gptq_gemm_gcu(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    bit: int,
    bias=None,
    group_size=128,
) -> torch.Tensor:
    assert bit in [4, 8]

    if bit == 4:
        out_shape = a.shape[:-1] + (b_q_weight.shape[-1],)
        reshaped_a = a.reshape(-1, a.shape[-1])

        output = torch.ops._C.gptq_gemm_gcu(
            reshaped_a,
            b_q_weight,
            b_gptq_qzeros,
            b_gptq_scales,
            b_g_idx,
            bit,
            bias,
            group_size,
        )
        return output.reshape(out_shape)
    elif bit == 8:
        out_shape = a.shape[:-1] + (b_q_weight.shape[0],)
        output = torch.empty(out_shape, dtype=a.dtype, device=a.device)
        torch.ops._C.weight_only_quant(
            output, a, b_q_weight, bias, b_gptq_scales, group_size
        )
        return output


# 8bit
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp is None
        ), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, None)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales)
    return output, input_scales, input_azp


def scaled_int8_dequant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=scale.dtype)
    if scale is not None:
        # static-per-tensor quantization.
        torch.ops._C.static_scaled_int8_dequant(output, input, scale)
        return output, scale
    else:
        assert False, "dynamic scaled int8 quant not support yet"


def gelu_tanh_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    torch.ops._C.gelu_tanh_quant(out, input, scale)


def gelu_quant(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor) -> None:
    torch.ops._C.gelu_quant(out, input, scale)


def gelu_new_quant(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor) -> None:
    torch.ops._C.gelu_new_quant(out, input, scale)


def silu_quant(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor) -> None:
    torch.ops._C.silu_quant(out, input, scale)


def gelu_fast_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    torch.ops._C.gelu_fast_quant(out, input, scale)


def gelu_tanh_asym_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor
) -> None:
    torch.ops._C.gelu_tanh_asym_quant(out, input, scale, qzero)


def gelu_asym_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor
) -> None:
    torch.ops._C.gelu_asym_quant(out, input, scale, qzero)


def gelu_new_asym_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor
) -> None:
    torch.ops._C.gelu_new_asym_quant(out, input, scale, qzero)


def silu_asym_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor
) -> None:
    torch.ops._C.silu_asym_quant(out, input, scale, qzero)


def gelu_fast_asym_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor
) -> None:
    torch.ops._C.gelu_fast_asym_quant(out, input, scale, qzero)


def rms_norm_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    scaling: torch.Tensor,
) -> None:
    torch.ops._C.rms_norm_quant(output, input, weight, scaling, epsilon)


def fused_add_rms_norm_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    scaling: torch.Tensor,
) -> None:
    torch.ops._C.fused_add_rms_norm_quant(
        output, input, residual, weight, epsilon, scaling
    )


def silu_mul_quant(
    out: torch.Tensor, input: torch.Tensor, scaling: torch.Tensor
) -> None:
    torch.ops._C.silu_mul_quant(out, input, scaling)


def gelu_mul_quant(
    out: torch.Tensor, input: torch.Tensor, scaling: torch.Tensor
) -> None:
    torch.ops._C.gelu_mul_quant(out, input, scaling)


def gelu_tanh_mul_quant(
    out: torch.Tensor, input: torch.Tensor, scaling: torch.Tensor
) -> None:
    torch.ops._C.gelu_tanh_mul_quant(out, input, scaling)


def layer_norm_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    scaling: torch.Tensor,
) -> None:
    torch.ops._C.layer_norm_quant(
        output, input, scaling, normalized_shape, weight, bias, epsilon
    )


def dot_bias_quant(
    out: torch.Tensor,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    torch.ops._C.dot_bias_quant(out, lhs, rhs, scale, bias)


def dispatch_bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    indicies: torch.Tensor,
    layer_idx: int,
    scale: float,
):
    w = w.unsqueeze(1)
    torch.ops._C.dispatch_bgmv(y, x, w, indicies, layer_idx, scale)


# dispatch bgmv
def dispatch_bgmv_low_level(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    indicies: torch.Tensor,
    layer_idx: int,
    scale: float,
    h_in: int,
    h_out: int,
    y_offset: int,
):
    w = w.unsqueeze(1)
    torch.ops._C.dispatch_bgmv_low_level(
        y, x, w, indicies, layer_idx, scale, h_in, h_out, y_offset
    )


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    x_q = torch.empty_like(x, device=x.device, dtype=dtype)

    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    torch.ops._C.dynamic_per_token_group_fp8_quant(x_q, x_s, x, group_size)

    return x_q, x_s


# fused moe
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
    )


def sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    torch.ops._moe_C.sgl_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
    )


def moe_align_block_size_pad(
    topk_ids,
    topk_ids_size,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
):
    torch.ops._C.moe_align_block_size_pad(
        topk_ids,
        topk_ids_size,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
    )


def get_ep_indices(
    ep_count,
    ep_token_indices,
    ep_valid_token_indices,
    topk_ids,
    expert_per_rank,
    ep_size,
):
    torch.ops._C.get_ep_indices(
        ep_count,
        ep_token_indices,
        ep_valid_token_indices,
        topk_ids,
        expert_per_rank,
        ep_size,
    )


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    N, _ = B.shape
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    torch.ops._C.linear_quant(C, A, B, bias, As, Bs)

    return C
