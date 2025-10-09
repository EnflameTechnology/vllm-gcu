from vllm_gcu.utils import is_vllm_equal, is_torch_equal_or_newer, is_nixl_equal


if is_vllm_equal("0.8.0"):
    from vllm_gcu.patch import patch_0_8_0  # noqa
elif is_vllm_equal("0.9.2"):
    from vllm_gcu.patch import patch_0_9_2  # noqa
elif is_vllm_equal("0.10.2"):
    from vllm_gcu.patch import patch_0_10_2  # noqa
elif is_vllm_equal("0.11.0"):
    from vllm_gcu.patch import patch_0_11_0  # noqa

if is_torch_equal_or_newer("2.6"):
    from vllm_gcu.patch import torch_2_6_0  # noqa

if is_nixl_equal('0.5.0'):
    from vllm_gcu.patch import nixl_0_5_0