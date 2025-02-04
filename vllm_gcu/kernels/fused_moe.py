#!/usr/bin/env python
# coding=utf-8

import functools
from typing import Any, Callable, Dict, List, Optional

import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_default_config,
    grouped_topk,
    moe_align_block_size,
)
from vllm.utils import vllm_lib

from vllm_gcu.kernels import _custom_ops as ops

logger = init_logger(__name__)


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
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    use_int8_w8a8: bool,
    block_shape: Optional[List[int]] = None,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            A, A_scale = ops.scaled_fp8_quant(A, A_scale)
        else:
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            A, A_scale = ops.per_token_group_quant_fp8(A, block_k)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape and block_shape[0] == 0
        assert B_zp is not None or B_zp.ndim == 3
    elif use_int8_w8a8:
        assert A_scale is not None
        assert B_scale is not None
    else:
        assert A_scale is None
        assert B_scale is None

    EM = sorted_token_ids.shape[0]
    block_size = config["BLOCK_SIZE_M"]
    if A.shape[0] < block_size:
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * block_size)

    if use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16 or use_int8_w8a8:
        if use_int8_w8a8:
            B_zp = None
            group_size = 1
        elif use_int4_w4a16:
            A_scale = None
            group_size = block_shape[1]

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
            None,
        )


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
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
        use_fp8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
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
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
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
        use_fp8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
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
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    if use_fp8_w8a8 and use_int8_w8a16:
        # hack int8_w8a8 when both True
        use_int8_w8a8 = True
        use_fp8_w8a8 = False
        use_int8_w8a16 = False
    else:
        use_int8_w8a8 = False

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    get_config_func = functools.partial(
        get_default_config,
        E=w2.shape[0],
        N=w2.shape[2],
        K=w1.shape[2],
        topk=top_k_num,
        dtype="",
        is_marlin=False,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    intermediate_cache1 = torch.empty(
        (M, top_k_num, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, top_k_num, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[: tokens_in_chunk * top_k_num]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            top_k_num,
            config,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a8=use_int8_w8a8,
            block_shape=block_shape,
        )

        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            a2_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a8=use_int8_w8a8,
            block_shape=block_shape,
        )
        # TODO: replace with moe_sum
        torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )
    return out_hidden_states


from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)


def forward_oot(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

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
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


UnquantizedFusedMoEMethod.forward_oot = forward_oot
