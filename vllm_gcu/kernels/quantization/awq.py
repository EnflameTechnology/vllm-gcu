from typing import List, Optional

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.awq import (
    AWQConfig,
    AWQLinearMethod,
    is_layer_skipped_awq,
)

from vllm_gcu.kernels import _custom_ops as ops

# from vllm.model_executor.layers.quantization import register_quantization_config
from vllm_gcu.kernels.quantization.utils import (
    register_gcu_quantization_config,
    register_weight_loader_v2_supported,
)


@register_gcu_quantization_config("awq")
class AWQGCUConfig(AWQConfig):
    def __repr__(self) -> str:
        return (
            f"AWQGCUConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point})"
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "awq_gcu"

    @classmethod
    def get_min_capability(cls) -> int:
        return 30

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["AWQGCULinearMethod"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQGCULinearMethod(self)
        return None

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if (
            "quant_method" in hf_quant_cfg
            and hf_quant_cfg["quant_method"] == "awq"
        ):
            return cls.get_name()
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


@register_weight_loader_v2_supported
class AWQGCULinearMethod(AWQLinearMethod):
    """Linear method for AWQ with GCU impl.

    Args:
        quant_config: The AWQGCU quantization config.
    """

    def __init__(self, quant_config: AWQGCUConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm_gcu.kernels.quantization.rearrange import (
            rearrange_uint4_int32_uint8_awq,
        )

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        qweight, qzeros = rearrange_uint4_int32_uint8_awq(
            self,
            qweight=layer.qweight.cpu(),
            qzeros=layer.qzeros.cpu(),
            scales=layer.scales.cpu(),
            rearrange_group=128,
            zeros_in_int8=True,
        )
        if layer.qweight.nbytes == qweight.nbytes:
            layer.qweight.data = layer.qweight.data.view(qweight.dtype).reshape(
                qweight.shape
            )
            layer.qweight.data.copy_(qweight.data)
        else:
            # TODO: reduce memory fragmentation
            layer.qweight.data = qweight.to(layer.qweight.device)
            torch.gcu.empty_cache()
        layer.qzeros.data = qzeros.to(layer.qweight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)

        out_shape = x.shape[:-1] + (qweight.shape[-1],)
        out = ops.awq_gemm_gcu(
            x,
            qweight,
            scales,
            qzeros,
            pack_factor,
            bias=bias,
            group_size=self.quant_config.group_size,
        )
        return out.reshape(out_shape)
