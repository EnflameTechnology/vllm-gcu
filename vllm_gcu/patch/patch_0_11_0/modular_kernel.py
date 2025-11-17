from typing import Optional, Union
import torch
from unittest.mock import patch

from vllm import envs
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    _chunk_scales, _moe_problem_size, prod, _resize_cache,
    count_expert_num_tokens, ExpertTokensMetadata, FusedMoEPrepareAndFinalize,
    FusedMoEPermuteExpertsUnpermute, SharedResizableBuffer)
from vllm.utils import cdiv
import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels.prepare_finalize import AlltoAllPrepareAndFinalize
from vllm.v1.worker.ubatching import (dbo_current_ubatch_id, dbo_enabled,
                                      dbo_maybe_run_recv_hook,
                                      dbo_register_recv_hook, dbo_yield)


class FusedMoEModularKernel(torch.nn.Module):
    """
    This class combines a FusedMoEPrepareAndFinalize instance and
    a FusedMoEPermuteExpertsUnpermute to provide an interface that
    is compatible with the `fused_experts` function in fused_moe.py.

    It takes care of managing any required scratch space.

    Note: Instances of this class should only be used for a single model
    layer due to any layer specific state that may be used by the component
    objects.
    """

    class SharedBuffers:

        def __init__(self) -> None:
            self.fused_out = SharedResizableBuffer()
            self.workspace13 = SharedResizableBuffer()
            self.workspace2 = SharedResizableBuffer()

    # Persistent buffers that are shared across `FusedMoEModularKernel`
    # instances (layers), to save memory and allocattions.
    #
    # We have two sets of buffers to support dual batch overlap (DBO) where each
    # microbatch (ubatch) should use its own set of buffers to avoid
    # cross-ubatch contimination.
    # NOTE that memory is lazily allocated for these buffers, meaning that if
    # DBO isn't being used, the second SharedBuffers will be empty.
    shared_buffers: list[SharedBuffers] = [SharedBuffers(), SharedBuffers()]

    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
        shared_experts: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        self.shared_experts = shared_experts
        if hasattr(self.prepare_finalize, 'set_shared_experts'):
            self.prepare_finalize.set_shared_experts(
                self.shared_experts)
        assert prepare_finalize.activation_format == \
            fused_experts.activation_formats[0], (
                f"{prepare_finalize.__class__.__name__}."
                f"{prepare_finalize.activation_format} == "
                f"{fused_experts.__class__.__name__}."
                f"{fused_experts.activation_formats[0]}")

    def _do_fused_experts(
        self,
        fused_out: Optional[torch.Tensor],
        a1: torch.Tensor,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:

        _, M, N, K, top_k = _moe_problem_size(a1q, w1, w2, topk_ids)

        (workspace13_shape, workspace2_shape, fused_out_shape,
         workspace_dtype) = self.fused_experts.workspace_shapes(
             a1, a1q, M, N, K, top_k, global_num_experts, local_num_experts,
             expert_tokens_meta)

        # We can reuse the memory between cache1 and cache3 because by the
        # time we need cache3, we're done with cache1.
        workspace13 = torch.empty(prod(workspace13_shape),
                                  device=a1.device,
                                  dtype=workspace_dtype)
        workspace2 = torch.empty(prod(workspace2_shape),
                                 device=a1.device,
                                 dtype=workspace_dtype)

        assert fused_out is None or fused_out.shape == fused_out_shape, (
            f"fused_out {fused_out.shape} but expected {fused_out_shape}")
        if fused_out is None:
            # reuse workspace13 for the output
            fused_out = _resize_cache(workspace13, fused_out_shape)

        self.fused_experts.apply(
            fused_out,
            a1q,
            w1,
            w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        return fused_out

    def _maybe_chunk_fused_experts(
        self,
        a1: torch.Tensor,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:

        _, M, N, K, top_k = _moe_problem_size(a1q, w1, w2, topk_ids)

        CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
        num_chunks = cdiv(M, CHUNK_SIZE)

        # TODO(bnell): get rid of one level here, update slice functions
        # to nops on num_chunks==1

        if not self.fused_experts.supports_chunking() or num_chunks == 1:
            return self._do_fused_experts(
                fused_out=None,
                a1=a1,
                a1q=a1q,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                a1q_scale=a1q_scale,
                a2_scale=self.fused_experts.a2_scale,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # Chunking required case
        assert num_chunks > 1

        # Construct the entire output that can then be processed in chunks.
        (_, _, fused_out_shape, _) = self.fused_experts.workspace_shapes(
            a1, a1q, M, N, K, top_k, global_num_experts, local_num_experts,
            expert_tokens_meta)
        ubatch_idx = dbo_current_ubatch_id()
        buffers = self.shared_buffers[ubatch_idx]
        fused_out = buffers.fused_out.get(fused_out_shape,
                                          device=a1q.device,
                                          dtype=a1.dtype)

        def slice_input_tensors(
            chunk_idx: int
        ) -> tuple[torch.Tensor, Optional[torch.Tensor],
                   Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
            s = chunk_idx * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, M)
            return (
                a1q[s:e],
                _chunk_scales(a1q_scale, s, e),
                _chunk_scales(self.fused_experts.a2_scale, s, e),
                topk_ids[s:e],
                topk_weights[s:e]
            )

        def slice_output_tensor(chunk_idx: int) -> torch.Tensor:
            assert fused_out.size(0) % M == 0, (
                f"fused_out shape {fused_out.shape} vs M {M}")
            factor = fused_out.size(0) // M
            out_chunk_size = CHUNK_SIZE * factor
            s = chunk_idx * out_chunk_size
            e = min(s + out_chunk_size, fused_out.size(0))
            return fused_out[s:e]

        def slice_expert_tokens_metadata(
                full_expert_tokens_meta: ExpertTokensMetadata,
                chunk_topk_ids: torch.Tensor, local_num_experts: int,
                expert_map: Optional[torch.Tensor]) -> ExpertTokensMetadata:
            # The existing expert_num_tokens is for the entire a1q
            # input. Chunking forces recomputation of the number
            # of tokens assigned to each expert.
            c_expert_num_tokens = count_expert_num_tokens(
                chunk_topk_ids, local_num_experts, expert_map)

            c_expert_num_tokens_cpu = None
            need_expert_num_tokens_cpu = (
                full_expert_tokens_meta.expert_num_tokens_cpu is not None)
            if need_expert_num_tokens_cpu:
                # This is blocking as some implementations need the count
                # on the CPU to determine appropriate input/out fused-moe
                # buffers
                c_expert_num_tokens_cpu = c_expert_num_tokens.to(
                    "cpu", non_blocking=False)

            return ExpertTokensMetadata(
                expert_num_tokens=c_expert_num_tokens,
                expert_num_tokens_cpu=c_expert_num_tokens_cpu)

        for chunk_idx in range(num_chunks):
            c_a1q, c_a1q_scale, c_a2_scale, c_topk_ids, c_topk_weights = (
                slice_input_tensors(chunk_idx))

            c_expert_tokens_meta = None
            if expert_tokens_meta is not None:
                if isinstance(self.prepare_finalize,
                              AlltoAllPrepareAndFinalize):
                    c_expert_num_tokens = None
                    if expert_tokens_meta.expert_num_tokens is not None:
                        c_expert_num_tokens = torch.clamp(
                            expert_tokens_meta.expert_num_tokens,
                            min=0,
                            max=CHUNK_SIZE)
                        expert_tokens_meta.expert_num_tokens -= CHUNK_SIZE
                    c_expert_tokens_meta = ExpertTokensMetadata(
                        expert_num_tokens=c_expert_num_tokens,
                        expert_num_tokens_cpu=None)
                else:
                    c_expert_tokens_meta = slice_expert_tokens_metadata(
                        expert_tokens_meta, c_topk_ids, local_num_experts,
                        expert_map)

            self._do_fused_experts(
                fused_out=slice_output_tensor(chunk_idx),
                a1=a1,
                a1q=c_a1q,
                w1=w1,
                w2=w2,
                topk_weights=c_topk_weights,
                topk_ids=c_topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                a1q_scale=c_a1q_scale,
                a2_scale=c_a2_scale,
                expert_tokens_meta=c_expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        return fused_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        This function computes a Mixture of Experts (MoE) layer using two sets
        of weights, w1 and w2, and top-k gating mechanism.

        Parameters:
        - hidden_states: (torch.Tensor): The input tensor to the MoE layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights (torch.Tensor): The topk weights applied at the end of
            the layer.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - inplace (bool): If True, perform the operation in-place.
            Defaults to False.
        - activation (str): The activation function to apply after the first
            MoE layer.
        - global_num_experts (int): The total number of experts in the global
            expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
            from the global expert space to the local expert space of the expert
            parallel shard.
        - w1_scale (Optional[torch.Tensor]): Optional scale to be used for w1.
        - w2_scale (Optional[torch.Tensor]): Optional scale to be used for w2.
        - w1_zp (Optional[torch.Tensor]): Optional zero points to be used for
            w1.
        - w2_zp (Optional[torch.Tensor]): Optional zero points to be used for
            w2.
        - a1_scale (Optional[torch.Tensor]): Optional scale to be used for a1.
        - a2_scale (Optional[torch.Tensor]): Optional scale to be used for a2.
        - apply_router_weight_on_input (bool): When true, the topk weights are
            applied directly on the inputs. This is only applicable when topk is
            1.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """

        if gcu_envs.VLLM_GCU_FORCE_EP_BALANCE:
            from vllm.distributed import get_ep_group
            ep_rank = get_ep_group().rank

            num_tokens, num_topk = topk_ids.shape
            local_num_experts = w1.size(0)
            ep_size = global_num_experts // local_num_experts

            num_tokens_across_ranks = get_ep_group().all_gather(
                torch.ones(1, device=topk_ids.device) * num_tokens, dim=0)
            token_start_loc = torch.zeros(ep_size + 1,
                                          device=topk_ids.device,
                                          dtype=topk_ids.dtype)
            token_start_loc[1:] = num_tokens_across_ranks.cumsum(dim=0)

            step = global_num_experts // num_topk
            base_expert_ids = torch.arange(0,
                                           global_num_experts,
                                           step,
                                           device=topk_ids.device,
                                           dtype=topk_ids.dtype)

            token_indices = torch.arange(num_tokens,
                                         device=topk_ids.device,
                                         dtype=topk_ids.dtype)

            row_offsets = (token_indices + token_start_loc[ep_rank]) % step

            topk_ids = base_expert_ids.unsqueeze(0) + row_offsets.unsqueeze(1)
            topk_ids = torch.remainder(topk_ids, global_num_experts)

        a1 = hidden_states

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        shared_output: Optional[torch.Tensor] = None

        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            assert not dbo_enabled()

            prepare_ret = self.prepare_finalize.prepare(
                 a1,
                 topk_weights,
                 topk_ids,
                 global_num_experts,
                 expert_map,
                 apply_router_weight_on_input,
                 self.fused_experts.quant_config,
             )
            if hasattr(self.prepare_finalize, 'set_shared_experts'):
                (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids,
                _expert_topk_weights, shared_output) = prepare_ret
            else:
                (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids,
                _expert_topk_weights) = prepare_ret
        else:
            # Overlap shared expert compute with all2all dispatch.
            dbo_maybe_run_recv_hook()
            prepare_ret = self.prepare_finalize.prepare_async(
                a1,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
            )

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = prepare_ret \
                if isinstance(prepare_ret, tuple) else (None, prepare_ret)

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids,
             _expert_topk_weights) = receiver()

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = (topk_weights if _expert_topk_weights is None else
                        _expert_topk_weights)

        fused_out = None

        if a1q.numel() == 0:
            # This happens when none of the tokens from the all2all reach this
            # EP rank. Also, note that this is only relevant for CUDAGraph
            # incompatible all2all kernels like the DeepEP high-throughput
            # kernels. CUDAGraph compatible all2all kernels like the pplx
            # kernels and the DeepEP low-latency kernels are always batched
            # and can never run into the tensor.numel() == 0 case.
            fused_out = torch.empty_like(a1q).to(dtype=a1.dtype)
        else:
            fused_out = self._maybe_chunk_fused_experts(
                a1=a1,
                a1q=a1q,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                a1q_scale=a1q_scale,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # NOTE: a1 and a1q might be same buffer with output if inplace
        if hasattr(self.prepare_finalize, 'set_shared_experts'):
            if shared_output is not None:
                output = a1.copy_(shared_output) if inplace else shared_output
            else:
                output = a1.fill_(0) if inplace else torch.zeros_like(a1)
            del a1, a1q
        else:
            output = torch.empty_like(a1)
            if envs.VLLM_ALL2ALL_BACKEND not in [
                "deepep_high_throughput", "deepep_low_latency"
            ]:
                output.fill_(0)

        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()

            self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            if self.shared_experts is not None and shared_output is None:
                shared_output = self.shared_experts(a1)
        else:
            finalize_ret = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = finalize_ret \
                if isinstance(finalize_ret, tuple) else (None, finalize_ret)

            enable_parallel_compute = gcu_envs.VLLM_GCU_ENABLE_PARALLEL_COMPUTE

            if enable_parallel_compute and self.shared_experts is not None and shared_output is None:
                shared_output = self.shared_experts(a1)

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            receiver()
            if not enable_parallel_compute and self.shared_experts is not None and shared_output is None:
                shared_output = self.shared_experts(a1)

        if not hasattr(self.prepare_finalize, 'set_shared_experts') and self.shared_experts is not None:
            output.add_(shared_output)

        if self.shared_experts is None:
            return output
        else:
            return shared_output, output


# yapf:disable
patch("vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEModularKernel", FusedMoEModularKernel).start()
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoEModularKernel", FusedMoEModularKernel).start()
