from unittest.mock import patch
import torch
from vllm_gcu.kernels._custom_ops import merge_attn_states
from vllm._custom_ops import reshape_and_cache_flash
try:
    from flash_attn.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


def get_flash_attn_version(requires_alibi: bool = False):
    return 2


def flash_attn_supports_fp8() -> bool:
    return True


# WA for https://github.com/vllm-project/vllm/pull/10989
def reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
) -> None:

    key = key[:slot_mapping.shape[0]]
    value = value[:slot_mapping.shape[0]]
    torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache,
                                                   value_cache, slot_mapping,
                                                   kv_cache_dtype, k_scale,
                                                   v_scale)


patch("vllm.attention.utils.fa_utils.get_flash_attn_version", get_flash_attn_version).start()
patch("vllm.attention.backends.flash_attn.get_flash_attn_version", get_flash_attn_version).start()
patch("vllm.attention.backends.flash_attn.flash_attn_supports_fp8", flash_attn_supports_fp8).start()
patch("vllm.attention.backends.flash_attn.flash_attn_varlen_func", flash_attn_varlen_func).start()
patch("vllm.attention.backends.flash_attn.flash_attn_with_kvcache", flash_attn_with_kvcache).start()

patch("vllm.v1.attention.backends.flash_attn.get_flash_attn_version", get_flash_attn_version).start()
patch("vllm.v1.attention.backends.flash_attn.flash_attn_supports_fp8", flash_attn_supports_fp8).start()
patch("vllm.v1.attention.backends.flash_attn.merge_attn_states", merge_attn_states).start()

import vllm.v1.attention.backends.flash_attn  # noqa
setattr(vllm.v1.attention.backends.flash_attn, "flash_attn_varlen_func", flash_attn_varlen_func)
setattr(vllm.v1.attention.backends.flash_attn, "reshape_and_cache_flash", reshape_and_cache_flash)
