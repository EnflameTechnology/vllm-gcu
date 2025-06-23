from vllm_gcu.utils import is_vllm_equal


if is_vllm_equal("0.8.0"):
    from vllm_gcu.patch import patch_0_8_0  # noqa
elif is_vllm_equal("0.9.1"):
    from vllm_gcu.patch import patch_0_9_1  # noqa
