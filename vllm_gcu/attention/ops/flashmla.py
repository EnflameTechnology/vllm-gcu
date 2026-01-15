from typing import Optional, Tuple
import torch


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize),
                                 torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax.
                       Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        descale_q: (batch_size), torch.float32. Descaling factors for Q.
        descale_k: (batch_size), torch.float32. Descaling factors for K.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    out, softmax_lse = torch.ops._flashmla_C.fwd_kvcache_mla(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse


def flash_mla_with_kvcache_sparse(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
    - q: (batch_size, seq_len_q, num_heads_q, head_dim).
    - k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
    - block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
    - cache_seqlens: (batch_size), torch.int32.
    - head_dim_v: Head dimension of v.
    - tile_scheduler_metadata:
        (num_sm_parts, TileSchedulerMetaDataSize), torch.int32,
        returned by get_mla_metadata.
    - num_splits:
        (batch_size + 1), torch.int32, returned by get_mla_metadata.
    - softmax_scale: float.
        The scale of QK^T before applying softmax.
        Default to 1 / sqrt(head_dim).
    - causal: bool. Whether to apply causal attention mask.
    - descale_q: (batch_size),
        torch.float32. Descaling factors for Q, used for fp8 quantization.
    - descale_k: (batch_size),
        torch.float32. Descaling factors for K, used for fp8 quantization.
    - is_fp8_kvcache: bool.
        Whether the k_cache and v_cache are in fp8 format.
        For the format of FP8 KV cache, please refer to README.md
    - indices: (batch_size, seq_len_q, topk), torch.int32.
        If not None, sparse attention will be enabled,
        and only tokens in the `indices` array will be attended to.
        Invalid indices should be set to -1 or numbers >= total_seq_len_kv.
        For details about how to set up `indices`, please refer to README.md.

    Returns:
    - out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
    - softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    if indices is not None:
        # NOTE (zyongye): sparse attention is also causal
        # since it only attend to the tokens before
        # but here `causal` should not be specified
        assert not causal, \
            "causal must be `false` if sparse attention is enabled."
    assert (descale_q is None) == (
        descale_k is None
    ), "descale_q and descale_k should be both None or both not None"

    if indices is None and q.element_size() == 1:
        out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
            q, k_cache, head_dim_v, cache_seqlens, block_table, softmax_scale,
            causal, tile_scheduler_metadata, num_splits, descale_q, descale_k)
    else:
        out, softmax_lse = torch.ops._flashmla_C.fwd_kvcache_mla_sparse(
            q, k_cache, head_dim_v, cache_seqlens, block_table, softmax_scale,
            causal, tile_scheduler_metadata, num_splits, is_fp8_kvcache, indices)
    return out, softmax_lse

# GCU version of get_mla_metadata. Not exactly the same as the original one.
def get_mla_metadata(
    tile_scheduler_metadata: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, None]:
    """
    Arguments:
    - tile_scheduler_metadata for GCU:
            (24 * 1024 * 1024), dtype torch.int8.
    - cache_seqlens: (batch_size), dtype torch.int32.
    """
    torch.ops._flashmla_C.get_mla_decoding_metadata(tile_scheduler_metadata,
                                                    cache_seqlens)