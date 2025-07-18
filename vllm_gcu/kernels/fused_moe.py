#!/usr/bin/env python
# coding=utf-8
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch_gcu
import vllm.envs as envs

from vllm.forward_context import ForwardContext, get_forward_context

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.utils import vllm_lib

import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels import _custom_ops as ops


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:

    # for gcu, block_size=64 can get large bandwidth
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    return config


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor = None,
    topk_ids_size=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    # sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    # expert_ids.fill_(0)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    if topk_ids_size is not None:
        if not gcu_envs.VLLM_GCU_DEEPSEEK_FUSION or expert_map is None:
            ops.moe_align_block_size_pad(
                topk_ids,
                topk_ids_size,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )
            if expert_map is not None:
                expert_ids = torch_gcu.gcu.efficient.gcu_index(expert_map, [expert_ids])
        else:
            torch.ops._C.exts_moe_align_block_size(
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                topk_ids,
                topk_ids_size,
                expert_map,
                num_experts,
                block_size,
            )
    else:
        if num_experts >= 224:
            if envs.VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON:
                raise NotImplementedError
            else:
                ops.sgl_moe_align_block_size(
                    topk_ids,
                    num_experts,
                    block_size,
                    sorted_ids,
                    expert_ids,
                    num_tokens_post_pad,
                )
        else:
            ops.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )

        if expert_map is not None:
            expert_ids = expert_map[expert_ids]
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: Optional[List[int]] = None,
    real_token_num=None,
    per_channel_quant=False,
    bias=None,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape and block_shape[0] == 0
        assert B_zp is None or B_zp.ndim == 3
    else:
        assert A_scale is None
        assert B_scale is None

    block_size = config["BLOCK_SIZE_M"]

    if use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16 or use_int8_w8a8:
        if use_fp8_w8a8:
            B_zp = None
            group_size = block_shape[1]
        elif use_int8_w8a8:
            B_zp = None
            group_size = 1
            if B_scale is not None and B_scale.ndim == 3:
                B_scale = B_scale.transpose(1,2).contiguous()
        elif use_int4_w4a16:
            A_scale = None
            group_size = block_shape[1]
        elif use_int8_w8a16:
            A_scale = None
            group_size = -1
        else:
            raise NotImplementedError

        torch.ops._C.fused_moe_quant_kernel(
            C,
            A,
            B,
            A_scale,
            B_scale,
            group_size,
            B_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            block_size,
            None,
            real_token_num,
        )
    else:
        topk_weights = topk_weights.to(torch.float32)  # WA for grouped_topk
        torch.ops._C.fused_moe_kernel(
            C,
            A,
            B,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            block_size,
            bias,
        )


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> None:

    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        True,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:

    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
    )


vllm_lib.impl(
    "inplace_fused_experts", inplace_fused_experts, dispatch_key="PrivateUse1"
)
vllm_lib.impl(
    "outplace_fused_experts", outplace_fused_experts, dispatch_key="PrivateUse1"
)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    b1=None,
    b2=None,
):
    from vllm_gcu.kernels.modular_experts import TritonExpertsPad
    from vllm_gcu.kernels.prepare_finalize import get_prepare_finalize
    from vllm_gcu.kernels.modular_kernel import FusedMoEModularKernel
    activation_and_layer_name = activation.split("_", 1)
    if len(activation_and_layer_name) > 1:
        activation, layer_name = activation_and_layer_name
    else:
        layer_name = None

    assert activation == "silu", f"not support activation: {activation}"

    forward_context: ForwardContext = get_forward_context()
    if layer_name is not None:
        layer = forward_context.no_compile_layers[layer_name]
        shared_experts = getattr(layer, "shared_experts", None)
        routed_scaling_factor = getattr(layer, "routed_scaling_factor", None)
        if shared_experts is not None:
            assert routed_scaling_factor is not None
    else:
        shared_experts = None
        routed_scaling_factor = 1.0

    use_ep = expert_map is not None
    prepare_finalize = get_prepare_finalize(use_ep, forward_context)
    prepare_finalize.set_shared_experts(shared_experts, routed_scaling_factor)
    fused_experts = TritonExpertsPad(use_fp8_w8a8, use_int8_w8a8,
                                     use_int8_w8a16, use_int4_w4a16, False,
                                     block_shape)
    modular = FusedMoEModularKernel(prepare_finalize=prepare_finalize,
                                    fused_experts=fused_experts)
    return modular(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace,
        activation,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        apply_router_weight_on_input,
    )


def forward_oot(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
) -> torch.Tensor:

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

    fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )

    return x


UnquantizedFusedMoEMethod.forward_oot = forward_oot


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    topk_weights = torch.empty(
        (gating_output.shape[0], topk), device=gating_output.device, dtype=torch.float32
    )
    topk_ids = torch.empty(
        (gating_output.shape[0], topk), device=gating_output.device, dtype=torch.int32
    )

    if hidden_states.numel() == 0:
        return topk_weights, topk_ids

    torch.ops._C.fused_grouped_topk(
        topk_weights,
        topk_ids,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        e_score_correction_bias,
        scoring_func,
    )
    return topk_weights, topk_ids


if m := sys.modules.get("vllm.model_executor.layers.fused_moe", None):
    m.grouped_topk = grouped_topk
if m := sys.modules.get("vllm.model_executor.layers.fused_moe.fused_moe", None):
    m.grouped_topk = grouped_topk
if m := sys.modules.get("vllm.model_executor.layers.fused_moe.layer", None):
    m.grouped_topk = grouped_topk
