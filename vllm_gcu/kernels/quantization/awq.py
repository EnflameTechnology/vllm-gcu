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
            and user_quant == "awq"
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
        self.processed = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.processed:
            return
        from vllm_gcu.kernels.quantization.rearrange import (
            rearrange_uint4_int32_uint8_awq,
        )

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # from vllm.config import get_current_vllm_config
        # import vllm_gcu.envs as gcu_envs
        # model_config = get_current_vllm_config().model_config
        # parallel_config = get_current_vllm_config().parallel_config

        # if model_config.hf_text_config.model_type in ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp') \
        #         and parallel_config.enable_expert_parallel:
        #     zeros_in_int8 = True
        # else:
        #     zeros_in_int8 = False
        zeros_in_int8 = False

        qweight, qzeros = rearrange_uint4_int32_uint8_awq(
            self,
            qweight=layer.qweight.cpu(),
            qzeros=layer.qzeros.cpu(),
            scales=layer.scales.cpu(),
            rearrange_group=128,
            zeros_in_int8=zeros_in_int8
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

        self.processed = True

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
