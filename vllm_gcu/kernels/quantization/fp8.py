from typing import List, Optional, Callable

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.platforms import current_platform

from vllm.utils import vllm_lib

from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.kernels.quantization.utils import (
    register_gcu_quantization_config,
    register_weight_loader_v2_supported,
)
from vllm_gcu.kernels.modular_experts import TritonExpertsPad


@register_gcu_quantization_config("fp8")
class Fp8GCUConfig(Fp8Config):
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8GCULinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8GCUMoEMethod(self, layer)
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
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return apply_w8a8_block_fp8_linear(
                input=x.view(self.out_dtype) if x.dtype != self.out_dtype else x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale if x_scale is None else x_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            )
        else:
            return super().apply(layer, x, bias)


class Fp8GCUMoEMethod(Fp8MoEMethod):

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer,
    ):
        return TritonExpertsPad(self.moe_quant_config)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)

        zero_expert_num = getattr(layer, 'zero_expert_num', 0)
        zero_expert_type = getattr(layer, 'zero_expert_type', None)

        select_result = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
        )
        topk_weights, topk_ids, zero_expert_result = select_result

        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
        )


def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_block_fp8_supported: bool = False,
    use_aiter_and_is_supported: bool = False,
) -> torch.Tensor:
    output_dtype = input.dtype

    if input_scale is None:
        # input_2d = input.view(-1, input.shape[-1])
        q_input, x_scale = ops.per_token_group_quant_fp8(
            input,  # input_2d
            block_size[1],
            dtype=current_platform.fp8_dtype(),
            column_major_scales=False,
        )
    else:
        input = input.view(current_platform.fp8_dtype())
        q_input = input
        x_scale = input_scale

    output_shape = [*input.shape[:-1], weight.shape[0]]
    output = ops.w8a8_block_fp8_matmul(
        q_input,
        weight,
        x_scale,
        weight_scale,
        block_size,
        output_dtype=output_dtype,
        bias=None,
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


# vllm_lib.impl(
#     "apply_w8a8_block_fp8_linear",
#     apply_w8a8_block_fp8_linear,
#     dispatch_key="PrivateUse1",
# )
