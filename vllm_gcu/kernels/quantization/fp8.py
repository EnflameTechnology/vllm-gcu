from typing import List, Optional

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.platforms import current_platform

from vllm.utils import vllm_lib

from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.kernels.quantization.utils import (
    register_gcu_quantization_config,
    register_weight_loader_v2_supported,
)


@register_gcu_quantization_config("fp8")
class Fp8GCUConfig(Fp8Config):
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8GCULinearMethod(self)
        return super().get_quant_method(layer, prefix)

    @classmethod
    def get_name(cls) -> str:
        return "fp8_gcu"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if (
            "quant_method" in hf_quant_cfg
            and hf_quant_cfg["quant_method"] == "fp8"
            and user_quant in ["fp8", "fp8_gcu", None]
        ):
            return cls.get_name()
        return None


@register_weight_loader_v2_supported
class Fp8GCULinearMethod(Fp8LinearMethod):
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            )
        else:
            return super().apply(layer, x, bias)


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

    # input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    q_input, x_scale = ops.per_token_group_quant_fp8(
        input,  # input_2d
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
