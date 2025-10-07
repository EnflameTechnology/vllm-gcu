from unittest.mock import patch
from vllm_gcu.kernels._custom_ops import per_token_group_quant_fp8
from vllm._custom_ops import scaled_int8_quant
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


def per_token_quant_int8_gcu(x):
    x_q, scales, _ = scaled_int8_quant(x)
    return x_q, scales

def moe_kernel_quantize_input_gcu(A,
    A_scale,
    quant_dtype,
    per_act_token_quant,
    block_shape=None,
    is_fp4_scale_swizzled=True
):
    # for wa8a-int8-gcu , to get the  A and A_scale of the input tensor
    if A_scale is not None and A.dtype == A_scale.dtype:
        return A, A_scale
    else:
        return moe_kernel_quantize_input(A, A_scale, quant_dtype, per_act_token_quant, block_shape, is_fp4_scale_swizzled)

patch(
    "vllm.model_executor.layers.quantization.utils.fp8_utils.per_token_group_quant_fp8",
    per_token_group_quant_fp8).start()
patch("vllm.model_executor.layers.fused_moe.utils.per_token_group_quant_fp8",
      per_token_group_quant_fp8).start()

patch("vllm.model_executor.layers.fused_moe.utils.per_token_quant_int8",
      per_token_quant_int8_gcu).start()
patch("vllm.model_executor.layers.fused_moe.prepare_finalize.moe_kernel_quantize_input",
      moe_kernel_quantize_input_gcu).start()
