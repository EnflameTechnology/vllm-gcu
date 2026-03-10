import torch
from unittest.mock import patch
from typing import Optional, Callable, Union

from vllm.distributed.parallel_state import get_ep_group
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase, FusedMoEPrepareAndFinalize, FusedMoE)
from vllm.model_executor.layers.fused_moe.prepare_finalize import MoEPrepareAndFinalizeNoEP
from vllm.forward_context import get_forward_context
from vllm.utils import cdiv

from vllm.logger import init_logger
from vllm.platforms import current_platform

import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels.prepare_finalize import AlltoAllSelector
from vllm_gcu.kernels.modular_experts import TritonExpertsPad
from vllm_gcu.patch.patch_0_11_0.modular_kernel import FusedMoEModularKernel
from vllm_gcu.kernels._custom_ops import eplb_map_to_physical_and_record

logger = init_logger(__name__)

origin_make = FusedMoEMethodBase.maybe_make_prepare_finalize


def init_prepare_finalize(self, layer: torch.nn.Module):
    assert self.moe is not None

    # We must get the quant config here so that the layer is
    # completely initialized, i.e. all weights loaded and post
    # processed.
    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    # We allow no ep use modular kernel.
    world_size = get_ep_group().world_size

    prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None

    # NOTE: use origin_make after deepep gcu supports fp8
    if self.moe.use_deepep_ll_kernels:
        from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import DeepEPLLPrepareAndFinalize, DEEPEP_QUANT_BLOCK_SHAPE
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert all2all_manager is not None
        assert self.moe_quant_config is not None
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=self.moe.max_num_tokens,
            token_hidden_size=self.moe.hidden_dim,
            num_ep_ranks=all2all_manager.world_size,
            num_global_experts=self.moe.num_experts,
            num_local_experts=self.moe.num_experts //
            all2all_manager.world_size)
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            self.moe_quant_config.quant_dtype == current_platform.fp8_dtype()
            and self.moe_quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE)
        use_fp8_dispatch &= gcu_envs.VLLM_GCU_DEEPEP_USE_FP8_DISPATCH

        prepare_finalize = DeepEPLLPrepareAndFinalize(
            handle,
            max_tokens_per_rank=self.moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
        )
    elif self.moe.use_deepep_ht_kernels:
        from vllm_gcu.kernels.deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalizeGCU
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert self.moe.dp_size == all2all_manager.dp_world_size

        all_to_all_args = dict()
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = DeepEPHTPrepareAndFinalizeGCU(
            handle,
            num_dispatchers=all2all_manager.world_size,
            dp_size=all2all_manager.dp_world_size,
            rank_expert_offset=all2all_manager.rank *
            self.moe.num_local_experts,
        )
    elif (self.moe.use_pplx_kernels or self.moe.use_deepep_ht_kernels
          or self.moe.use_deepep_ll_kernels):
        prepare_finalize = origin_make(self)
    elif self.moe.moe_parallel_config.ep_size > 1 and (
            self.moe.moe_parallel_config.dp_size > 1
            or gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL):
        # DP*SP -> EP
        prepare_finalize = AlltoAllSelector(
            None, self.moe.moe_parallel_config.dp_size)
    else:
        # TP -> EP || 1 card
        prepare_finalize = MoEPrepareAndFinalizeNoEP()

    if prepare_finalize is not None:
        logger.debug("%s for %s(%s)", prepare_finalize.__class__.__name__,
                        self, id(self))
        assert self.topk_indices_dtype is None
        assert self.fused_experts is None, \
            f"Attempt to override experts for {id(self)}!"
        self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
        experts = self.select_gemm_impl(prepare_finalize, layer)
        if (self.moe.moe_parallel_config.ep_size == 1
                and self.moe.moe_parallel_config.dp_size > 1):
            self.fused_experts = FusedMoEModularKernel(
                prepare_finalize,
                experts,
            )
        else:
            add_shared = getattr(layer, 'add_shared', False)
            self.fused_experts = FusedMoEModularKernel(
                prepare_finalize,
                experts,
                layer.shared_experts,
                add_shared,
            )


def select_gemm_impl_unquant(
    self,
    prepare_finalize,
    layer,
):
    return TritonExpertsPad(self.moe_quant_config)


origin_forward_impl = FusedMoE.forward_impl


def forward_impl(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    shared_output = None
    if (self.shared_experts is not None
            and self.moe_parallel_config.ep_size == 1
            and self.moe_parallel_config.dp_size > 1):
        origin_shared_experts = self.shared_experts
        shared_output = self.shared_experts(hidden_states)
        self._shared_experts = None

    final_hidden_states = origin_forward_impl(self, hidden_states,
                                              router_logits)

    if shared_output is not None:
        self._shared_experts = origin_shared_experts
        if getattr(self, 'add_shared', False):
            final_hidden_states.add_(shared_output)
        return shared_output, final_hidden_states
    return final_hidden_states


def forward_impl_chunked(
    self,
    full_hidden_states: torch.Tensor,
    full_router_logits: torch.Tensor,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    assert self.batched_hidden_states is not None
    assert self.batched_router_logits is not None
    assert self.batched_hidden_states.dtype == full_hidden_states.dtype
    assert self.batched_router_logits.dtype == full_router_logits.dtype
    # Check size compatibility.
    assert (
        self.batched_hidden_states.size(-1) == full_hidden_states.size(-1))
    assert (
        self.batched_router_logits.size(-1) == full_router_logits.size(-1))

    self.ensure_moe_quant_config()

    full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
    if self.shared_experts is not None:
        full_shared_final_hidden_states = torch.empty_like(
            full_hidden_states)

    def process_chunk(chunk_start, chunk_end, skip_result_store=False):
        chunk_size = chunk_end - chunk_start
        hidden_states = full_hidden_states[chunk_start:chunk_end, :]
        router_logits = full_router_logits[chunk_start:chunk_end, :]

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            enable_eplb=self.enable_eplb,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
        )

        assert self.shared_experts is None or isinstance(
            final_hidden_states, tuple)

        if self.zero_expert_num is not None and self.zero_expert_num > 0:
            assert isinstance(final_hidden_states, tuple)
            assert self.shared_experts is None
            final_hidden_states, zero_expert_result = final_hidden_states
            if zero_expert_result is not None:
                final_hidden_states += zero_expert_result

        if not skip_result_store:
            if self.shared_experts is None:
                full_fused_final_hidden_states[
                    chunk_start:chunk_end, :].copy_(final_hidden_states,
                                                    non_blocking=True)
            else:
                full_shared_final_hidden_states[
                    chunk_start:chunk_end, :].copy_(final_hidden_states[0],
                                                    non_blocking=True)
                full_fused_final_hidden_states[
                    chunk_start:chunk_end, :].copy_(final_hidden_states[1],
                                                    non_blocking=True)
        return final_hidden_states

    ctx = get_forward_context()
    # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
    max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
    moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

    # If the input to the MoE is sequence parallel then divide by sp_size
    # to find the maximum number of tokens for any individual dispatcher.
    if self.is_sequence_parallel:
        max_tokens_across_dispatchers = cdiv(max_tokens_across_dispatchers,
                                                self.sp_size)

    if max_tokens_across_dispatchers <= moe_dp_chunk_size_per_rank:
        return process_chunk(0,
                             max_tokens_across_dispatchers,
                             skip_result_store=True)
    num_tokens = full_hidden_states.size(0)
    for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers,
                    moe_dp_chunk_size_per_rank)):
        chunk_start = chunk_start_
        chunk_end = min(chunk_start + moe_dp_chunk_size_per_rank,
                        max_tokens_across_dispatchers)
        # clamp start and end
        chunk_start = max(0, min(chunk_start, num_tokens - 1))
        chunk_end = min(chunk_end, num_tokens)
        with ctx.dp_metadata.chunked_sizes(self.sp_size,
                                            moe_dp_chunk_size_per_rank,
                                            chunk_idx):
            process_chunk(chunk_start,
                            chunk_end,
                            skip_result_store=chunk_start_ >= num_tokens)

    if self.shared_experts is None:
        return full_fused_final_hidden_states
    else:
        return (full_shared_final_hidden_states,
                full_fused_final_hidden_states)

def forward_cuda(
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
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

    zero_expert_num = getattr(layer, 'zero_expert_num', 0)
    zero_expert_type = getattr(layer, 'zero_expert_type', None)

    topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
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
        zero_expert_type=zero_expert_type)

    if self.rocm_aiter_moe_enabled:
        assert self.fused_experts is None
        result = self.rocm_aiter_fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input)
    elif self.flashinfer_cutlass_moe_enabled:
        return self.flashinfer_cutlass_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input)
    elif self.fused_experts is not None:
        result = self.fused_experts(
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
    else:
        from vllm.model_executor.layers.fused_moe import fused_experts
        assert fused_experts is not None
        result = fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            quant_config=self.moe_quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    if zero_expert_num != 0 and zero_expert_type is not None:
        assert not isinstance(result, tuple), \
            "Shared + zero experts are mutually exclusive not yet supported"
        return result, zero_expert_result
    else:
        return result

# yapf: disable
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoEMethodBase.init_prepare_finalize", init_prepare_finalize).start()
patch("vllm.model_executor.layers.fused_moe.FusedMoEMethodBase.init_prepare_finalize", init_prepare_finalize).start()
patch("vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.select_gemm_impl", select_gemm_impl_unquant).start()
patch("vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.forward_cuda", forward_cuda).start()
patch("vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.forward_native", forward_cuda).start()
patch("vllm.model_executor.layers.fused_moe.layer.eplb_map_to_physical_and_record", eplb_map_to_physical_and_record).start()
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_impl_chunked", forward_impl_chunked).start()
patch("vllm.model_executor.layers.fused_moe.FusedMoE.forward_impl_chunked", forward_impl_chunked).start()
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_impl", forward_impl).start()
patch("vllm.model_executor.layers.fused_moe.FusedMoE.forward_impl", forward_impl).start()
