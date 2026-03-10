import pytest
import torch
import vllm_gcu._C


def ref_flashmla_mixed(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    head_size_v: int,
    seqlens_k: torch.Tensor,
    tile_scheduler_metadata,
    num_splits,
    is_fp8_kvcache,
    indices: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    descale_k: torch.Tensor,
    threshold: int,
    cu_seq_q: torch.Tensor,
) -> torch.Tensor:
    num_seqs = len(cu_seq_q) - 1
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size_k = kv_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = (cu_seq_q[i + 1] - cu_seq_q[i]).item()
        kv_len = seqlens_k[i]
        q = query[start_idx : start_idx + query_len]
        q *= softmax_scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        if kv_len <= threshold:
            kv = kv_cache[block_indices].view(-1, num_kv_heads, head_size_k)
            kv = kv[:kv_len]
            if is_fp8_kvcache:
                kv = kv.to(torch.bfloat16) * descale_k.to(torch.bfloat16)
            k = kv
            v = kv[..., :head_size_v]
            if q.shape[1] != k.shape[1]:
                k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
                v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
            attn = torch.einsum("qhd,khd->hqk", q, k).float()
            empty_mask = torch.ones(query_len, kv_len)
            if is_causal:
                mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
                attn.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1).to(v.dtype)
            out = torch.einsum("hqk,khd->qhd", attn, v)
        else:
            ind = indices[start_idx : start_idx + query_len]
            kv = kv_cache.view(-1, num_kv_heads, head_size_k)[ind]
            if is_fp8_kvcache:
                kv = kv.to(torch.bfloat16) * descale_k.to(torch.bfloat16)
            k = kv
            v = kv[..., :head_size_v]
            if q.shape[1] != k.shape[2]:
                k = torch.repeat_interleave(k, q.shape[1] // k.shape[2], dim=2)
                v = torch.repeat_interleave(v, q.shape[1] // v.shape[2], dim=2)
            attn = torch.einsum("qhd,qthd->hqt", q, k).float()
            attn = torch.softmax(attn, dim=-1).to(v.dtype)
            out = torch.einsum("hqt,qthd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 4243)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads_q", [128, 32, 16])
@pytest.mark.parametrize("threshold", [0, 2048, 8192])
def test_flashmla_mixed(
    seq_lens,
    num_heads_q,
    threshold,
    head_dim_k=576,
    head_dim_v=512,
    num_heads_k=1,
    block_size=64,
    bytes_per_token=576,
    topk=2048,
    num_blocks=2048,
):
    torch.set_default_device("gcu")
    batch_size = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    query = torch.randn(sum(query_lens), num_heads_q, head_dim_k, dtype=torch.bfloat16)
    k_cache = torch.randn(
        (num_blocks, block_size, num_heads_k, bytes_per_token),
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_table = torch.randint(
        0, num_blocks, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    indices = torch.zeros((sum(query_lens), topk), dtype=torch.int32)
    q_seq_per_hk = sum(query_lens) * num_heads_q // num_heads_k
    tile_scheduler_metadata = torch.empty(
        24 * 1024 * 1024,
        dtype=torch.int8,
    )
    k_scale = torch.tensor([1.0], dtype=torch.float32)
    torch.ops._flashmla_C.get_mla_decoding_metadata(
        tile_scheduler_metadata,
        kv_lens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q,
        True,
        topk,
        threshold,
        cu_query_lens,
    )
    _attn_out, _ = torch.ops._flashmla_C.fwd_kvcache_mla_mixed(
        q=query,
        kcache=k_cache,
        block_table=block_table,
        head_size_v=head_dim_v,
        seqlens_k=kv_lens,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=None,
        is_fp8_kvcache=True,
        indices=indices,
        softmax_scale=head_dim_k**-0.5,
        is_causal=True,
        descale_k=k_scale,
        threshold=threshold,
        cu_seq_q=cu_query_lens,
    )

    torch.set_default_device("cpu")
    ref_out = ref_flashmla_mixed(
        query.cpu(),
        k_cache.to(torch.bfloat16).cpu(),
        block_table.cpu(),
        head_dim_v,
        kv_lens.cpu(),
        tile_scheduler_metadata=None,
        num_splits=None,
        is_fp8_kvcache=True,
        indices=indices.cpu(),
        softmax_scale=head_dim_k**-0.5,
        is_causal=True,
        descale_k=k_scale.cpu(),
        threshold=threshold,
        cu_seq_q=cu_query_lens.cpu(),
    )

    assert torch.allclose(_attn_out.cpu(), ref_out, rtol=1e-1, atol=1e-1)
