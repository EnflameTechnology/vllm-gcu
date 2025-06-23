from unittest.mock import patch

try:
    from flash_attn.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


def get_flash_attn_version():
    return 2


patch("vllm.attention.backends.flash_attn.flash_attn_varlen_func", flash_attn_varlen_func).start()
patch("vllm.attention.backends.flash_attn.flash_attn_with_kvcache", flash_attn_with_kvcache).start()
patch("vllm.attention.backends.flash_attn.get_flash_attn_version", get_flash_attn_version).start()
