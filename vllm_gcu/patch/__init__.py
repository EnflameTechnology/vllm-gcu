from vllm.utils import is_torch_equal_or_newer
from vllm_gcu.utils import is_vllm_equal, is_nixl_equal


if is_nixl_equal('0.5.0'):
    # NOTE: before imported in nixl_connector patcher
    from vllm_gcu.patch import nixl_0_5_0

if is_vllm_equal("0.8.0"):
    from vllm_gcu.patch import patch_0_8_0  # noqa
elif is_vllm_equal("0.9.2"):
    from vllm_gcu.patch import patch_0_9_2  # noqa
elif is_vllm_equal("0.11.0"):
    from vllm_gcu.patch import patch_0_11_0  # noqa

if is_torch_equal_or_newer("2.6"):
    from vllm_gcu.patch import torch_2_6_0  # noqa

from vllm_gcu.patch import patch_lmcache
