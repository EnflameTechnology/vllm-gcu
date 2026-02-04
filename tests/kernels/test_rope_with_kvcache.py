import pytest
import torch
import torch_gcu
import random

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm_gcu.attention.backends.mla_v1_fusion import RopeWithKVCache
from vllm import _custom_ops as ops


def _dequantize_fp8_ds_mla_entry(
    cache_slice: torch.Tensor, kv_lora_rank: int, rope_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a single fp8_ds_mla cache entry back to latent + rope."""

    # The first kv_lora_rank bytes store FP8 latent values with one scale per
    # 128 element tile written as float32 right after the latent payload.
    scales = cache_slice.view(torch.float32)[kv_lora_rank // 4 : kv_lora_rank // 4 + 4]
    latent = torch.empty(kv_lora_rank, dtype=torch.float16, device=cache_slice.device)
    for tile_idx in range(4):
        tile_start = tile_idx * 128
        tile_end = tile_start + 128
        ops.convert_fp8(
            latent[tile_start:tile_end],
            cache_slice[tile_start:tile_end],
            float(scales[tile_idx].item()),
            kv_dtype="fp8",
        )
    latent = latent.to(dtype)

    rope_offset = kv_lora_rank // 2 + 8
    rope_vals = cache_slice.view(dtype)[rope_offset : rope_offset + rope_dim]
    return latent, rope_vals.clone()


@pytest.mark.parametrize("b", [4])
@pytest.mark.parametrize("s_q", [2])
@pytest.mark.parametrize("mean_sk", [4096])
@pytest.mark.parametrize("h_q", [128])
@pytest.mark.parametrize("h_kv", [1])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_ds_mla"])
def test_rope_with_kvcache(
    b,
    s_q,
    mean_sk,
    h_q,
    h_kv,
    block_size,
    kv_cache_dtype,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    kv_lora_rank=512,
    rms_norm_eps=1e-6,
    kv_scale=1.0,
):
    # TODO: parametrize using pytest
    dtype = torch.bfloat16
    device = torch.device("gcu:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.gcu.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}")

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)

    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = max_seqlen // 256 * 256

    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)

    if kv_cache_dtype == "fp8_ds_mla":
        kv_cache = torch.randn(
            block_table.numel(),
            block_size,
            kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2,
            device="cpu",
        ).to(torch.float8_e4m3fn)
    elif kv_cache_dtype == "auto":
        kv_cache = torch.randn(
            block_table.numel(),
            block_size,
            kv_lora_rank + qk_rope_head_dim,
            device="cpu",
        )
    else:
        assert "not impl"
    num_slots = block_size * block_table.numel()
    slot_mapping_lst = random.sample(range(num_slots), b * s_q)

    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)
    for i in range(b):
        kv_cache.view(b, max_seqlen_pad, h_kv, -1)[i, cache_seqlens[i].item() :] = (
            float("nan")
        )
    kv_cache = kv_cache.gcu()
    if kv_cache_dtype == "fp8_ds_mla":
        kv_cache = kv_cache.view(torch.uint8)
    kv_cache_ref = kv_cache.clone()

    q = torch.randn((b * s_q, h_q, qk_nope_head_dim + qk_rope_head_dim))
    q_ref = q.clone()
    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_pe_out = q_pe
    kv_c_and_k_pe = torch.randn((b * s_q, kv_lora_rank + qk_rope_head_dim))
    kv_c_and_k_pe_ref = kv_c_and_k_pe.clone()
    positions = torch.randint(1, max_seqlen, (b * s_q,), dtype=torch.int32)
    kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
    rope_scaling = {
        "rope_type": "deepseek_yarn",
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    }
    rotary_emb = get_rope(
        qk_rope_head_dim,
        rotary_dim=qk_rope_head_dim,
        max_position=max_seqlen,
        base=10000,
        rope_scaling=rope_scaling,
        is_neox_style=False,
    )
    kv_scale = torch.tensor(kv_scale, dtype=torch.float32)
    rope_with_kvcache = RopeWithKVCache(
        rotary_emb, kv_a_layernorm, kv_lora_rank, qk_rope_head_dim, kv_cache_dtype
    )

    rope_with_kvcache.forward_oot(
        q_pe_out, None, q_pe, kv_c_and_k_pe, kv_cache, slot_mapping, positions, kv_scale
    )

    q_nope_ref, q_pe_ref = q_ref.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_pe_out_ref = q_pe_ref
    rope_with_kvcache.forward_native(
        q_pe_out_ref,
        None,
        q_pe_ref,
        kv_c_and_k_pe_ref,
        kv_cache_ref,
        slot_mapping,
        positions,
        kv_scale,
    )
    assert torch.allclose(q_pe_out, q_pe_out_ref, 1e-2, 1e-2)
    if kv_cache_dtype == "auto":
        assert torch.allclose(kv_cache, kv_cache_ref, 1e-2, 1e-2)
    elif kv_cache_dtype == "fp8_ds_mla":
        for token_idx in range(b * s_q):
            slot = slot_mapping[token_idx].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            cache_slice = kv_cache[block_idx, block_offset]
            latent, rope_vals = _dequantize_fp8_ds_mla_entry(
                cache_slice, kv_lora_rank, qk_rope_head_dim, kv_c_and_k_pe.dtype
            )
            cache_slice = kv_cache_ref[block_idx, block_offset]
            latent_ref, rope_vals_ref = _dequantize_fp8_ds_mla_entry(
                cache_slice, kv_lora_rank, qk_rope_head_dim, kv_c_and_k_pe.dtype
            )
            assert torch.allclose(rope_vals, rope_vals_ref, 1e-2, 1e-2)
            assert torch.allclose(latent, latent_ref, 1e-1, 1e-1)
    else:
        assert "not impl"