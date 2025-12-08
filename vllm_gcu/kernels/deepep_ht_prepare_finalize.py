# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional, Union

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
    DeepEPHTPrepareAndFinalize)
import vllm_gcu.envs as gcu_envs


class DeepEPHTPrepareAndFinalizeGCU(DeepEPHTPrepareAndFinalize):

    def _receiver(
        self,
        event: deep_ep.EventOverlap,
        has_scales: bool,
        token_data: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        expert_topk_ids: Optional[torch.Tensor],
        num_experts: int,
        expert_num_tokens_per_expert_list: list[int],
        expert_topk_weights: Optional[torch.Tensor],
        a1_scale: Optional[torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        if event.event is not None:
            event.current_stream_wait()

        if has_scales:
            expert_x, expert_x_scale = token_data
        else:
            expert_x, expert_x_scale = token_data, None

        # The existing MOE kernels assume that all entries of topk_ids are
        # valid. To that effect, set the -1s in expert_topk_ids to some expert
        # outside this rank so the expert_map can remap it to -1 when safe.
        # With Expert Parallel, the experts are divided amongst the rank
        # sequentially. For rank 0, set it to num_experts - 1 and for all other
        # ranks set it to 0 as we know that expert_map will have a -1 in those
        # regions for those ranks.
        #
        # DeepEP's topk_ids output refers to the local experts directly. Offset
        # the topk_ids to move it back to the global experts space so it aligns
        # with existing vLLM interfaces.
        assert expert_topk_ids is not None
        expert_topk_ids = torch.where(
            expert_topk_ids == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            expert_topk_ids + self.rank_expert_offset)

        # Makes a GPU-CPU copy.
        # TODO (varun): Maybe it is better to re-compute the expert_num_tokens
        # on GPU.
        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            expert_num_tokens_per_expert_list, device=expert_x.device)

        # Dispatch and Quant
        # DeepEP kernels only support dispatching block-quantized
        # activation scales.
        # Dispatch in bfloat16 and quantize afterwards
        if not (gcu_envs.VLLM_GCU_DEEPEP_USE_FP8_DISPATCH and quant_config.is_block_quantized):
            # Quantize after dispatch.
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape)

        return (expert_x, expert_x_scale, expert_tokens_meta, expert_topk_ids,
                expert_topk_weights)

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.ReceiverType:

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        if gcu_envs.VLLM_GCU_DEEPEP_USE_FP8_DISPATCH and quant_config.is_block_quantized:
            # Quant and Dispatch
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = quant_config.a1_scale

        return self._do_dispatch(tokens=a1q,
                                 token_scales=a1q_scale,
                                 rank_topk_ids=topk_ids,
                                 rank_topk_weights=topk_weights,
                                 num_experts=num_experts,
                                 a1_scale=a1_post_scale,
                                 quant_config=quant_config)
