#!/usr/bin/env python
# coding=utf-8
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch_gcu
import vllm.envs as envs

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts # do not del
)
from vllm.model_executor.layers.fused_moe import FusedMoE

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
    A_scale_rec=None,
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
            if B.dtype != torch.int8:
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
        if use_fp8_w8a8 and B.dtype == torch.int8:
            # w4a8-fp8
            assert A_scale_rec is not None
            torch.ops._C.fused_moe_quant_kernel_ex(
                C,
                A,
                B,
                A_scale_rec, # vllm_gcu w4a8 input_scale_rep
                B_scale,
                B_zp,
                None, # bias
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                real_token_num,
                mul_routed_weight,
                top_k,
                block_size,
                128,
                -1,
            )
        else:
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
