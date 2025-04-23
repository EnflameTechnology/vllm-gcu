from vllm_gcu.utils import vllm_version_equal


if vllm_version_equal("0.8.0"):
    from vllm_gcu.patch.worker import patch_0_8_0  # noqa