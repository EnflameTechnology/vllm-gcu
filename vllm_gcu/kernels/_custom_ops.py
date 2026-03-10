from typing import List, Optional, Tuple

import tops_extension.torch  # noqa: F401
import torch
import torch_gcu  # noqa: F401

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

import vllm
import vllm_gcu._C  # noqa: F401
from vllm.vllm_flash_attn.flash_attn_interface import maybe_contiguous

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
    query_scales: Optional[torch.Tensor] = None,
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
        query_scales
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

# mrope-interleave ops
def mrotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    mrope_section: List[int],
    mrope_interleaved: bool,
) -> None:
    if query.numel() == 0:
        return

    torch.ops._C.mrotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox, mrope_section, mrope_interleaved
    )

# layer norm ops
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


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
        torch.ops._C.linear_quant(
            output, a, b_q_weight, bias, b_gptq_scales, None, group_size
        )
        return output

@register_fake("_C::gptq_gemm_gcu")
def _gptq_gemm_gcu_fake(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    bit: int,
    bias=None,
    group_size=128,
) -> torch.Tensor:
    out_shape = a.shape[:-1] + (b_q_weight.shape[-1],)
    return torch.empty(out_shape, dtype=a.dtype, device=a.device)



# 8bit
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        if use_per_token_if_dynamic:
            output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
            # dynamic-per-token quantization.
            shape = input.shape[:-1] + (1,)
            scale = torch.empty(shape, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub=scale_ub)
        else:
            raise NotImplementedError("dynamic_scaled_fp8_quant is not implemented for per tensor!")
    else:
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)
    return output, scale


def gelu_tanh_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    torch.ops._C.gelu_tanh_static_int8_quant(out, input, scale)


def gelu_quant(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor) -> None:
    torch.ops._C.gelu_static_int8_quant(out, input, scale)


def gelu_new_quant(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor) -> None:
    torch.ops._C.gelu_static_int8_quant(out, input, scale)


def silu_quant(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor) -> None:
    torch.ops._C.silu_static_int8_quant(out, input, scale)


def gelu_fast_quant(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    torch.ops._C.gelu_fast_static_int8_quant(out, input, scale)


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
    torch.ops._C.rms_norm_static_int8_quant(output, input, weight, scaling, epsilon)


def fused_add_rms_norm_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    scaling: torch.Tensor,
) -> None:
    torch.ops._C.fused_add_rms_norm_static_int8_quant(
        output, input, residual, weight, epsilon, scaling
    )


def silu_mul_quant(
    out: torch.Tensor, input: torch.Tensor, scaling: torch.Tensor
) -> None:
    torch.ops._C.silu_mul_static_int8_quant(out, input, scaling)


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
    torch.ops._C.layer_norm_static_int8_quant(
        output, input, scaling, normalized_shape, weight, bias, epsilon
    )

def dispatch_bgmv(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    scale: float = 1.0,
):
    w = w.unsqueeze(1)
    torch.ops._C.dispatch_bgmv(y, x, w, indices, 0, scale)


def dispatch_bgmv_low_level(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    slice_offset: int,
    slice_size: int,
):
    w = w.unsqueeze(1)
    h_in = x.size(1)
    torch.ops._C.dispatch_bgmv_low_level(
        y, x, w, indices, 0, 1.0, h_in, slice_size, slice_offset
    )

def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    real_token_num: Optional[torch.Tensor] = None,
    use_ue8m0: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    x_q = torch.empty_like(x, device=x.device, dtype=dtype)

    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    if use_ue8m0 is not None:
        torch.ops._C.dynamic_per_token_group_fp8_quant_with_ue8m0(
            x_q, x_s, x, group_size, use_ue8m0
        )
    else:
        if real_token_num is None:
            torch.ops._C.dynamic_per_token_group_fp8_quant(x_q, x_s, x, group_size)
        else:
            torch.ops._C.dynamic_per_token_group_fp8_quant_with_size(
                x_q, x_s, x, real_token_num, group_size
            )

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
    group_size: int = -1
) -> torch.Tensor:
    N, _ = B.shape
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    torch.ops._C.linear_quant(C, A, B, bias, As, Bs, group_size)

    return C


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
):
    torch.ops._C.merge_attn_states(
        output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse
    )

#torch.ops._C.cutlass_scaled_mm.default.tags.append(torch._C.Tag.flexible_layout)

def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        indices_type: Optional[torch.dtype] = None) -> torch.Tensor:
    '''
    Map the logical expert ids to physical expert ids
    and record the expert load metrics.
    This will select a pseudo-random replica for each logical expert.
    Only used for EPLB.
    Args:
        topk_ids: The logical expert ids.
        expert_load_view: The expert load view.
        logical_to_physical_map: The logical to physical map.
        logical_replica_count: The logical replica count.
        indices_type: The indices type.
    Returns:
        The physical expert ids.
    '''
    if indices_type is not None:
        out = torch.empty_like(topk_ids, dtype=indices_type)
    else:
        out = torch.empty_like(topk_ids)

    num_redundant_experts = expert_load_view.shape[0] - logical_replica_count.shape[0]
    torch.ops._C.eplb_map_to_physical_and_record(
        out,
        topk_ids,
        expert_load_view,
        logical_to_physical_map[..., :num_redundant_experts + 1].contiguous(),
        logical_replica_count
    )

    return out

def indexer_k_quant_and_cache(k: torch.Tensor, kv_cache: torch.Tensor,
                              slot_mapping: torch.Tensor,
                              quant_block_size: int,
                              kv_cache_dtype: str) -> None:
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping,
                                                     quant_block_size,
                                                     kv_cache_dtype)


get_token_bin_counts_and_mask_origin = \
    vllm.model_executor.layers.utils.get_token_bin_counts_and_mask

def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
    return_bin_counts: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 3.0 not support get_token_bin_counts_and_mask fusion
    from vllm.platforms import current_platform
    if not current_platform.has_device_capability(140):
        return get_token_bin_counts_and_mask_origin(tokens, vocab_size, num_seqs)
    else:
        bin_counts = torch.empty((num_seqs, vocab_size),
                                dtype=torch.long,
                                device=tokens.device)
        mask = torch.empty((num_seqs, vocab_size),
                        dtype=torch.bool,
                        device=tokens.device)
        torch.ops._C.get_token_bin_counts_and_mask(bin_counts,
                                                        mask,
                                                        tokens,
                                                        vocab_size,
                                                        num_seqs,
                                                        return_bin_counts)
        return bin_counts, mask

def topk_softmax_renormalize(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indices: torch.Tensor, gating_output: torch.Tensor,
                 renormalize: bool) -> None:
    torch.ops._moe_C.topk_softmax_renormalize(topk_weights, topk_ids, token_expert_indices,
                                  gating_output, renormalize)


def reshape_and_cache_flash_int8kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_zp: torch.Tensor,
    v_zp: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash_int8kv(key, value, key_cache,
                                                   value_cache, slot_mapping,
                                                   kv_cache_dtype, k_scale,
                                                   v_scale, k_zp, v_zp)

DEFAULT_FA_VERSION = 2
def flash_attn_varlen_func_int8kv(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None, # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = None,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    k_zp=None,
    v_zp=None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "seqused_k must be provided if block_table is provided"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    # dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)

    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
                raise NotImplementedError(
                    "FA2 does not support scheduler_metadata, q_descale, "
                    "k_descale, v_descale"
                )
        if s_aux is not None:
            raise NotImplementedError("FA2 does not support s_aux")
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q, k, v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            None,
        )
    elif fa_version == 3:
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        out, softmax_lse, _, _ = torch.ops._C.mha_fwd_int8kv(
            q, k, v,
            None, None,       # k_new, v_new
            q_v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,     # cu_seqlens_k
            None,             # cu_seqlens_k_new
            None, seqused_k,  # seqused_q, seqused_k
            max_seqlen_q, max_seqlen_k,
            block_table,
            None,             # kv_batch_idx
            None,             # leftpad_k
            None, None, None, # rotary_cos, rotary_sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            k_zp, v_zp,
            softmax_scale,
            causal,
            real_window_size[0], real_window_size[1],
            softcap,
            True,             # rotary_interleaved
            scheduler_metadata,
            num_splits,
            None,             # pack_gqa
            0,                # sm_margin
            s_aux             # s_aux
        )
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    return (out, softmax_lse) if return_softmax_lse else out
