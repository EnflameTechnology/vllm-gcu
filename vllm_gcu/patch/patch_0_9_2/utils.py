from unittest.mock import patch
from vllm_gcu.kernels._custom_ops import per_token_group_quant_fp8


def tops_device_count():
    import torch_gcu  # noqa

    return torch_gcu._C._gcu_getDeviceCount()

patch("vllm.utils.cuda_device_count_stateless", tops_device_count).start()
patch("vllm.model_executor.layers.quantization.utils.fp8_utils.per_token_group_quant_fp8", per_token_group_quant_fp8).start()
patch("vllm.model_executor.layers.fused_moe.utils.per_token_group_quant_fp8", per_token_group_quant_fp8).start()
