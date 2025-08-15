from unittest.mock import patch

try:
    from flash_attn.vllm_flash_attn import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

def get_mla_metadata(
    cache_seqlens,
    num_heads_per_head_k,
    num_heads_k,
):
    return None, None

patch("vllm.v1.attention.backends.mla.common.flash_attn_varlen_func", flash_attn_varlen_func).start()
patch("vllm.v1.attention.backends.mla.flashmla.get_mla_metadata", get_mla_metadata).start()