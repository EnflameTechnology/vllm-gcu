from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.kernels.quantization.kv_cache import GCUBaseKVCacheMethod
from vllm_gcu.kernels.quantization.utils import register_gcu_quantization_config


@register_gcu_quantization_config("w8a8")
class W8A8Config(QuantizationConfig):
    """Config class for W8A8"""

    def __init__(
        self,
        group_size: int,
    ) -> None:
        # todo
        self.group_size = group_size

    def __repr__(self) -> str:
        return f"W8A8Config()"

    @classmethod
    def get_name(cls) -> str:
        return "w8a8_gcu"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 30

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W8A8Config":
        # todo
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(group_size)

    def get_linear_method(self) -> "W8A8LinearMethod":
        return W8A8LinearMethod(self)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["W8A8LinearMethod"]:
        from vllm.attention.layer import Attention

        if isinstance(layer, LinearBase):
            return W8A8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return FusedW8A8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Int8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if "quant_method" in hf_quant_cfg and hf_quant_cfg["quant_method"] == "w8a8":
            return cls.get_name()
        return None


class W8A8LinearMethod(LinearMethodBase):
    """Linear method for W8A8.

    Args:
        quant_config: The W8A8 quantization config.
    """

    def __init__(self, quant_config: W8A8Config):
        self.quant_config = quant_config
        self.processed = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> Dict[str, Any]:
        layer.out_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        # (N, K)
        qweight = Parameter(
            torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            requires_grad=False,
        )
        # (N)
        out_scales = Parameter(
            torch.ones(output_size_per_partition, dtype=torch.float32),
            requires_grad=False,
        )
        # (K)
        # TODO: this can be optimized as an optional param,
        # like what maybe_w8a8() does.
        # leave it here now to leverage linear tp policy.
        in_scales = Parameter(
            torch.ones(input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(out_scales, {"output_dim": 0})
        set_weight_attrs(in_scales, {"input_dim": 0, "ignore_warning": True})

        layer.register_parameter("weight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("out_scales", out_scales)
        set_weight_attrs(out_scales, extra_weight_attrs)
        layer.register_parameter("in_scales", in_scales)
        set_weight_attrs(in_scales, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.processed:
            return

        w = layer.weight.T.contiguous()
        layer.weight.resize_(w.shape)
        layer.weight.data.copy_(w.data)
        layer.in_scales.data.reciprocal_()

        self.processed = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dtype in self.quant_config.get_supported_act_dtypes():
            x = ops.scaled_int8_quant(x, layer.in_scales)[0]
        assert x.dtype == torch.int8

        shape = list(x.shape)
        shape[-1] = int(layer.weight.shape[-1])
        output = torch.empty(shape, dtype=layer.out_dtype, device=x.device)
        ops.dot_bias_quant(output, x, layer.weight, layer.out_scales, bias)
        return output


class FusedW8A8MoEMethod(FusedMoEMethodBase):
    """fuded moe method for W8A8.

    Args:
        quant_config: The W8A8 quantization config.
    """

    def __init__(self, quant_config: W8A8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> Dict[str, Any]:
        int8_dtype = torch.int8

        strategy = FusedMoeWeightScaleSupported.CHANNEL.value
        # scales_size13 = 1
        # scales_size2 = 1

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": False})

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=int8_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=int8_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_out_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_out_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_in_scales", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, intermediate_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_in_scales", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        b1=None,
        b2=None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.fused_w8a8_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=True,
            use_int8_w8a16=True,
            use_int4_w4a16=False,
            w1_scale=layer.w13_out_scales,
            w2_scale=layer.w2_out_scales,
            w1_zp=None,
            w2_zp=None,
            a1_scale=layer.w13_in_scales,
            a2_scale=layer.w2_in_scales,
        )


class Int8KVCacheMethod(GCUBaseKVCacheMethod):
    def __init__(self, quant_config: W8A8Config):
        super().__init__(quant_config)
