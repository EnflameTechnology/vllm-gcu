from unittest.mock import patch
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


patch("vllm.attention.ops.merge_attn_states.merge_attn_states", merge_attn_states).start()
patch("vllm.attention.utils.fa_utils.get_flash_attn_version", get_flash_attn_version).start()
patch("vllm.attention.backends.flash_attn.get_flash_attn_version", get_flash_attn_version).start()
patch("vllm.attention.backends.flash_attn.flash_attn_supports_fp8", flash_attn_supports_fp8).start()
patch("vllm.attention.backends.flash_attn.flash_attn_varlen_func", flash_attn_varlen_func).start()
patch("vllm.attention.backends.flash_attn.flash_attn_with_kvcache", flash_attn_with_kvcache).start()

patch("vllm.v1.attention.backends.flash_attn.get_flash_attn_version", get_flash_attn_version).start()
patch("vllm.v1.attention.backends.flash_attn.flash_attn_supports_fp8", flash_attn_supports_fp8).start()
patch("vllm.v1.attention.backends.flash_attn.merge_attn_states", merge_attn_states).start()
patch("vllm.v1.attention.backends.mla.common.merge_attn_states", merge_attn_states).start()

import vllm.v1.attention.backends.flash_attn  # noqa
setattr(vllm.v1.attention.backends.flash_attn, "flash_attn_varlen_func", flash_attn_varlen_func)
setattr(vllm.v1.attention.backends.flash_attn, "reshape_and_cache_flash", reshape_and_cache_flash)
