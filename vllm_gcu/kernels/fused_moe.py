#!/usr/bin/env python
# coding=utf-8
import functools
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch_gcu
import vllm.envs as envs
from vllm.config import get_current_vllm_config

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    get_default_config,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.utils import vllm_lib
from vllm.distributed.parallel_state import get_tp_group

from vllm_gcu.distributed.parallel_state import all_to_all_v2
from vllm_gcu.kernels import _custom_ops as ops


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
        ops.moe_align_block_size_pad(
            topk_ids,
            topk_ids_size,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
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
        expert_ids = torch_gcu.gcu.efficient.gcu_index(expert_map, [expert_ids])
        # expert_ids = expert_map[expert_ids.to(torch.int64)]
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
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    use_int8_w8a8: bool,
    block_shape: Optional[List[int]] = None,
    real_token_num=None,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            A, A_scale = ops.scaled_fp8_quant(A, A_scale)
        else:
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = ops.per_token_group_quant_fp8(A, block_k)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape and block_shape[0] == 0
        assert B_zp is None or B_zp.ndim == 3
    elif use_int8_w8a8:
        assert A_scale is not None
        assert B_scale is not None
    else:
        assert A_scale is None
        assert B_scale is None

    block_size = config["BLOCK_SIZE_M"]

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
            None,
        )


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
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
        activation,
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
    activation: str = "silu",
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
        activation,
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
    activation: str = "silu",
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
    from vllm.distributed import get_world_group
    from vllm.forward_context import get_forward_context

    import vllm_gcu.envs as gcu_envs

    assert activation == "silu", f"not support activation: {activation}"

    parallel_config = get_current_vllm_config().parallel_config
    scheduler_config = get_current_vllm_config().scheduler_config
    recv_token_total = None
    if parallel_config.enable_expert_parallel:

        use_max = (
            not gcu_envs.VLLM_GCU_DEBUG_PDONLY
        ) or get_forward_context().attn_metadata.num_prefills > 0

        max_model_len = (
            scheduler_config.max_num_batched_tokens
            if use_max
            else scheduler_config.max_num_seqs
        )
        if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
            sp_size = get_tp_group().world_size
            max_model_len = (max_model_len + sp_size -1) // sp_size * sp_size

        group = get_world_group().device_group
        ep_size = get_world_group().world_size
        expert_per_rank = global_num_experts // ep_size

        hidden_states_ori = hidden_states
        ep_split_size = torch.empty(
            [ep_size], dtype=torch.int32, device=topk_ids.device
        )
        ep_token_indices = torch.empty(
            [hidden_states.shape[0] * topk_ids.shape[1]],
            dtype=torch.int32,
            device=topk_ids.device,
        )
        send_token_total = torch.empty([1], dtype=torch.int32, device=topk_ids.device)
        ops.get_ep_indices(
            ep_split_size,
            ep_token_indices,
            send_token_total,
            topk_ids,
            expert_per_rank,
            ep_size,
        )

        topk_ids_width = topk_ids.shape[1] * topk_ids.element_size()
        assert topk_ids_width % hidden_states.element_size() == 0
        topk_ids_width //= hidden_states.element_size()
        topk_weights_width = topk_weights.shape[1] * topk_weights.element_size()
        assert topk_weights_width % hidden_states.element_size() == 0
        topk_weights_width //= hidden_states.element_size()

        send_packed = torch.cat(
            (
                hidden_states,
                topk_ids.view(hidden_states.dtype),
                topk_weights.view(hidden_states.dtype),
            ),
            dim=1,
        )
        send_packed_sorted = torch_gcu.gcu.efficient.gcu_index(send_packed, [ep_token_indices])
        # send_packed_sorted = send_packed[ep_token_indices.to(torch.int64)]
        recv_packed = torch.empty(
            (
                max_model_len * parallel_config.data_parallel_size,
                hidden_states.shape[1] + topk_ids_width + topk_weights_width,
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        sp_split_size = torch.empty_like(ep_split_size)

        all_to_all_v2(
            recv_packed,
            send_packed_sorted,
            sp_split_size,
            ep_split_size,
            group=group,
            flag=1,
        )
        recv_token_total = torch.sum(sp_split_size, 0, True, dtype=torch.int32)

        # EP
        hidden_states = torch.empty(
            (recv_packed.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        topk_ids_ = torch.empty(
            (recv_packed.shape[0], topk_ids_width),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        topk_weights_ = torch.empty(
            (recv_packed.shape[0], topk_weights_width),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._C.dynamic_split(
            [hidden_states, topk_ids_, topk_weights_],
            recv_packed,
            recv_token_total.to(torch.uint32),
            [hidden_states.shape[1], topk_ids_width, topk_weights_width],
            1,
        )
        topk_ids = topk_ids_.view(topk_ids.dtype)
        topk_weights = topk_weights_.view(topk_weights.dtype)

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    # assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    # assert w2.is_contiguous(), "Expert weights2 must be contiguous"
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

    chunk_num = (num_tokens // CHUNK_SIZE) + 1
    for chunk in range(chunk_num):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
        if recv_token_total is not None:
            if chunk_num > 1:
                valid_in_chunk = torch.clamp(
                    recv_token_total, min=0, max=tokens_in_chunk
                )
                recv_token_total -= tokens_in_chunk
            else:
                valid_in_chunk = recv_token_total
        else:
            valid_in_chunk = torch.full(
                (1,),
                tokens_in_chunk,
                dtype=torch.int32,
                device=curr_hidden_states.device,
            )

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
            curr_topk_ids,
            config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
            valid_in_chunk,
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
            real_token_num=valid_in_chunk,
        )

        torch.ops._C.silu_and_mul_pad(
            intermediate_cache2.view(-1, top_k_num, N // 2),
            intermediate_cache1,
            valid_in_chunk,
        )

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
            real_token_num=valid_in_chunk,
        )
        # TODO: replace with moe_sum
        torch.ops._moe_C.moe_sum_pad(
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
            intermediate_cache3.view(*intermediate_cache3.shape),
            valid_in_chunk,
            1,
            False,
        )

    if parallel_config.enable_expert_parallel:
        sp_hidden_states = torch.zeros(
            (send_packed_sorted.shape[0], hidden_states.shape[1]),
            dtype=hidden_states_ori.dtype,
            device=hidden_states_ori.device,
        )
        all_to_all_v2(
            sp_hidden_states,
            out_hidden_states,
            ep_split_size,
            sp_split_size,
            group=group,
            flag=0,
        )
        if inplace:
            output = hidden_states_ori
            output.fill_(0)
        else:
            output = torch.zeros_like(hidden_states_ori)
        output.index_add_(
            0,
            ep_token_indices,
            sp_hidden_states,
        )
        return output
    else:
        return out_hidden_states


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
