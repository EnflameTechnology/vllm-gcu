from typing import List, Optional

import torch
from vllm.platforms import current_platform

from vllm.utils import vllm_lib

from vllm_gcu.kernels import _custom_ops as ops


def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_block_fp8_supported: bool = False,
) -> torch.Tensor:
    assert input_scale is None

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    q_input, x_scale = ops.per_token_group_quant_fp8(
        input_2d,
        block_size[1],
        dtype=current_platform.fp8_dtype(),
        column_major_scales=False,
    )
    output = ops.w8a8_block_fp8_matmul(
        q_input,
        weight,
        x_scale,
        weight_scale,
        block_size,
        output_dtype=input.dtype,
        bias=bias,
    )
    return output.to(dtype=input.dtype).view(*output_shape)


vllm_lib.impl(
    "apply_w8a8_block_fp8_linear",
    apply_w8a8_block_fp8_linear,
    dispatch_key="PrivateUse1",
)
