from unittest.mock import patch
from vllm_gcu.kernels._custom_ops import per_token_group_quant_fp8
from vllm._custom_ops import scaled_int8_quant


def tops_device_count():
    import torch_gcu  # noqa

    return torch_gcu._C._gcu_getDeviceCount()

def per_token_quant_int8_gcu(x):
    x_q, scales, _ = scaled_int8_quant(x)
    return x_q, scales


patch("vllm.utils.cuda_device_count_stateless", tops_device_count).start()
patch("vllm.model_executor.layers.quantization.utils.fp8_utils.per_token_group_quant_fp8", per_token_group_quant_fp8).start()
patch("vllm.model_executor.layers.fused_moe.utils.per_token_group_quant_fp8", per_token_group_quant_fp8).start()

patch("vllm.model_executor.layers.fused_moe.utils.per_token_quant_int8", per_token_quant_int8_gcu).start()