#!/usr/bin/env python
# coding=utf-8
import functools
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)

from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_default_config,
    grouped_topk,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.utils import vllm_lib

import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.distributed.all_to_all import all_to_all_v2

logger = init_logger(__name__)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor = None,
    topk_ids_size = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
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
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    use_int8_w8a8: bool,
    block_shape: Optional[List[int]] = None,
    real_token_num=None
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
        assert B_zp is None or B_zp.ndim == 3
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
    max_model_len=None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    from vllm.distributed import get_world_group

    recv_token_total = None
    if gcu_envs.VLLM_GCU_ENABLE_EXPERT_PARALLEL:
        assert max_model_len is not None
        dp_size = gcu_envs.VLLM_GCU_DATA_PARALLEL_SIZE
        group = get_world_group().device_group
        ep_size = get_world_group().world_size
        ep_rank = get_world_group().rank_in_group
        expert_per_rank = global_num_experts // ep_size

        hidden_states_ori = hidden_states
        ep_split_size = torch.empty([ep_size], dtype=torch.int32, device=topk_ids.device)
        ep_token_indices = torch.zeros([hidden_states.shape[0]*topk_ids.shape[1]], dtype=torch.int32, device=topk_ids.device)
        send_token_total = torch.empty([1], dtype=torch.int32, device=topk_ids.device)
        ops.get_ep_indices(ep_split_size, ep_token_indices, send_token_total, topk_ids, expert_per_rank, ep_size)
        # TODO:convert maybe loss acc, use view dtype
        send_packed = torch.cat(
            (
                hidden_states,
                topk_ids.to(hidden_states.dtype),
                topk_weights.to(hidden_states.dtype),
            ),
            dim=1,
        )
        send_packed_sorted = send_packed[ep_token_indices]
        recv_packed = torch.empty(
            (
                max_model_len * dp_size,
                hidden_states.shape[1] + topk_ids.shape[1] + topk_weights.shape[1],
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        sp_split_size = torch.empty_like(ep_split_size)
        all_to_all_v2(recv_packed, send_packed_sorted, sp_split_size, ep_split_size, group=group, flag=1)
        recv_token_total = sp_split_size.sum()

        ### EP ###
        hidden_states_, topk_ids_, topk_weights_ = torch.split(
            recv_packed, [hidden_states.shape[1], topk_ids.shape[1], topk_weights.shape[1]], dim=1)
        hidden_states = hidden_states_.contiguous()
        topk_ids = topk_ids_.to(topk_ids.dtype)
        topk_weights = topk_weights_.to(topk_weights.dtype)

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
        if recv_token_total is not None:
            valid_in_chunk = torch.clamp(recv_token_total, min=0, max=tokens_in_chunk).unsqueeze(0).to(torch.int32)
            recv_token_total -= tokens_in_chunk
        else:
            valid_in_chunk = torch.tensor([tokens_in_chunk], dtype=torch.int32, device=curr_hidden_states.device)

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
            curr_topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map,
            valid_in_chunk
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

        torch.ops._C.silu_and_mul_pad(intermediate_cache2, intermediate_cache1.view(-1, N), valid_in_chunk*top_k_num)

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
        torch.ops._moe_C.moe_sum_pad(out_hidden_states[begin_chunk_idx:end_chunk_idx],
                                     intermediate_cache3.view(*intermediate_cache3.shape),
                                     valid_in_chunk, 1, False)

    if gcu_envs.VLLM_GCU_ENABLE_EXPERT_PARALLEL:
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
            flag=0
        )

        ### SP ###
        output = torch.zeros_like(hidden_states_ori)
        output.index_add_(
            0,
            ep_token_indices.to(torch.int64),
            sp_hidden_states,
        )
        hidden_states_ori.copy_(output)
        return output
    else:
        return out_hidden_states


class EPFusedMoE(FusedMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        max_model_len=None
    ):

        self.global_num_experts = num_experts
        if gcu_envs.VLLM_GCU_ENABLE_EXPERT_PARALLEL:
            self.ep_size = get_world_group().world_size
            tp_size = 1
        else:
            self.ep_size = 1
            tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )

        if self.ep_size > 1:
            self.expert_map = torch.full(
                (self.global_num_experts,), -1, dtype=torch.int32
            )

            local_num_experts = num_experts // self.ep_size
            ep_rank = get_world_group().rank // tp_size
            if ep_rank < (self.ep_size - 1):
                self.expert_map[
                    ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts
                ] = torch.arange(0, local_num_experts, dtype=torch.int32)
            else:
                local_num_experts = num_experts - ep_rank * local_num_experts
                self.expert_map[-local_num_experts:] = torch.arange(
                    0, local_num_experts, dtype=torch.int32
                )
        else:
            self.expert_map = None
        self.max_model_len = max_model_len

        super().__init__(
            (
                torch.sum(self.expert_map != -1)
                if self.expert_map is not None
                else num_experts
            ),
            top_k,
            hidden_size,
            intermediate_size,
            params_dtype,
            reduce_results,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            quant_config,
            tp_size,
            prefix,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
        )
        self.tp_size = tp_size

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                self.quant_method.__class__.__name__
                == "CompressedTensorsWNA16MoEMethod"
            )
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]
        tp_rank = get_tensor_model_parallel_rank() % self.tp_size

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        # Case weight scales and zero_points
        if "scale" in weight_name or "zero" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        if isinstance(self.quant_method, UnquantizedFusedMoEMethod):
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias,
            )
        else:
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                max_model_len=self.max_model_len,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias,
            )

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states


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

    fused_experts_impl(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )

    return x


UnquantizedFusedMoEMethod.forward_oot = forward_oot


# This is used by the Deepseek-V2 and Deepseek-V3 model
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

    topk_weights = torch.zeros(
        (gating_output.shape[0], topk), device=gating_output.device, dtype=torch.float32
    )
    topk_ids = torch.zeros(
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


# [TODO] remove the monkey patch to fix official error
if m := sys.modules.get("vllm.model_executor.layers.fused_moe", None):
    m.grouped_topk = grouped_topk
if m := sys.modules.get("vllm.model_executor.layers.fused_moe.fused_moe", None):
    m.grouped_topk = grouped_topk
