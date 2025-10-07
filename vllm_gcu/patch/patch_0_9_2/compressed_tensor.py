from typing import Callable, Optional, Union
import torch
from unittest.mock import patch
from compressed_tensors.quantization import QuantizationStrategy

from vllm.model_executor.layers.fused_moe import FusedMoE

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsW8A8Int8MoEMethod
from vllm_gcu.kernels.modular_experts import TritonExpertsPad


class GCUCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):  # yapf: disable

    def select_gemm_impl(
        self,
        prepare_finalize,
        moe,
    ):
        return TritonExpertsPad(
            use_int8_w8a8=True,
            per_channel_quant=True,
            per_act_token_quant=self.input_quant.strategy ==
            QuantizationStrategy.TOKEN,
        )

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
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for "
                "`CompressedTensorsW8A8Int8MoEMethod` yet.")

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
            #routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias)
            #indices_type=self.topk_indices_dtype)

        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


patch(
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsW8A8Int8MoEMethod",
    GCUCompressedTensorsW8A8Int8MoEMethod).start()
