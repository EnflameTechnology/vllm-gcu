from vllm_gcu.utils import is_vllm_equal


if is_vllm_equal("0.8.0"):
    from vllm_gcu.patch.worker import patch_0_8_0  # noqa
