from unittest.mock import patch
from vllm_gcu.kernels._custom_ops import per_token_group_quant_fp8
from vllm._custom_ops import scaled_int8_quant
from vllm_gcu.utils import STR_DTYPE_TO_TORCH_DTYPE
from .arg_utils import _is_v1_supported_oracle


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
patch("vllm.utils.STR_DTYPE_TO_TORCH_DTYPE", STR_DTYPE_TO_TORCH_DTYPE).start()
patch("vllm.worker.cache_engine.STR_DTYPE_TO_TORCH_DTYPE", STR_DTYPE_TO_TORCH_DTYPE).start()
patch("vllm.v1.worker.gpu_model_runner.STR_DTYPE_TO_TORCH_DTYPE", STR_DTYPE_TO_TORCH_DTYPE).start()
patch("vllm.engine.arg_utils.EngineArgs._is_v1_supported_oracle", _is_v1_supported_oracle).start()
