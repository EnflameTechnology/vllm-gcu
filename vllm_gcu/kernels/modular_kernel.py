from math import prod
from typing import Optional
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.utils import cdiv
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize, FusedMoEPermuteExpertsUnpermute,
    _moe_problem_size, _chunk_scales)
import vllm_gcu.envs as gcu_envs


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

    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
    ):
        super().__init__()
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        assert prepare_finalize.activation_format == \
            fused_experts.activation_formats[0], (
                f"{prepare_finalize.__class__.__name__}."
                f"{prepare_finalize.activation_format} == "
                f"{fused_experts.__class__.__name__}."
                f"{fused_experts.activation_formats[0]}")

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
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        a1_scale_rec: Optional[torch.Tensor] = None,
        a2_scale_rec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        (a1q, a1q_scale, expert_num_tokens, _expert_topk_ids,
         _expert_topk_weights, shared_output) = self.prepare_finalize.prepare(
             a1,
             a1_scale,
             a2_scale,
             topk_weights,
             topk_ids,
             global_num_experts,
             expert_map,
             apply_router_weight_on_input,
             self.fused_experts.quant_config,
         )

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
            fused_out = torch.empty_like(a1q, dtype=a1.dtype)
        else:
            _, M, N, K, top_k = _moe_problem_size(a1q, w1, w2, topk_ids)

            if self.fused_experts.enable_chunking():
                CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
                num_chunks = cdiv(M, CHUNK_SIZE)
            else:
                CHUNK_SIZE = M
                num_chunks = 1

            if num_chunks == 1:
                (workspace13_shape, workspace2_shape, fused_out_shape,
                 workspace_dtype) = self.fused_experts.workspace_shapes(
                     a1, a1q, M, N, K, top_k, global_num_experts,
                     local_num_experts)
            else:
                # Use the full M to get the final output shape.
                _, _, fused_out_shape, _ = (
                    self.fused_experts.workspace_shapes(
                        a1, a1q, M, N, K, top_k, global_num_experts,
                        local_num_experts))
                # Use the CHUNK_SIZE to get the workspace shapes.
                workspace13_shape, workspace2_shape, _, workspace_dtype = (
                    self.fused_experts.workspace_shapes(
                        a1, a1q, CHUNK_SIZE, N, K, top_k, global_num_experts,
                        local_num_experts))

            # We can reuse the memory between cache1 and cache3 because by the
            # time we need cache3, we're done with cache1.
            workspace13 = torch.empty(prod(workspace13_shape),
                                      device=a1.device,
                                      dtype=workspace_dtype)
            workspace2 = torch.empty(prod(workspace2_shape),
                                     device=a1.device,
                                     dtype=workspace_dtype)

            if num_chunks == 1:
                fused_out = _resize_cache(workspace13, fused_out_shape)

                self.fused_experts.apply(
                    fused_out,
                    a1q,
                    w1,
                    w2,
                    topk_weights,  # vllm_gcu added for https://github.com/vllm-project/vllm/pull/20725
                    topk_ids,
                    activation=activation,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w1_zp=w1_zp,
                    w2_zp=w2_zp,
                    a1q_scale=a1q_scale,
                    a2_scale=a2_scale,
                    workspace13=workspace13,
                    workspace2=workspace2,
                    expert_num_tokens=expert_num_tokens,
                    apply_router_weight_on_input=
                    apply_router_weight_on_input,  # vllm_gcu added
                    a1q_scale_rec=a1_scale_rec,  # vllm_gcu for w4a8
                    a2_scale_rec=a2_scale_rec,  # vllm_gcu for w4a8
                )
            else:
                # The leading output dimension may not be equal to M, so
                # we compute output indices separately.
                M_out = fused_out_shape[0]
                assert M_out >= M
                factor = M_out // M
                assert factor > 0
                OUT_CHUNK_SIZE = CHUNK_SIZE * factor

                fused_out = torch.empty(fused_out_shape,
                                        device=a1q.device,
                                        dtype=workspace_dtype)

                assert cdiv(M_out, OUT_CHUNK_SIZE) == num_chunks, (
                    f"{cdiv(M_out, OUT_CHUNK_SIZE)} == {num_chunks}")

                for chunk in range(num_chunks):
                    begin_chunk_idx = chunk * CHUNK_SIZE
                    end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, M)
                    begin_out_idx = chunk * OUT_CHUNK_SIZE
                    end_out_idx = min((chunk + 1) * OUT_CHUNK_SIZE, M_out)
                    curr_a1q = a1q[begin_chunk_idx:end_chunk_idx]
                    curr_a1q_scale = _chunk_scales(a1q_scale, begin_chunk_idx,
                                                   end_chunk_idx)
                    curr_a2_scale = _chunk_scales(a2_scale, begin_chunk_idx,
                                                  end_chunk_idx)
                    curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
                    curr_topk_weights = topk_weights[
                        begin_chunk_idx:end_chunk_idx]
                    valid_in_chunk = None
                    if expert_num_tokens is not None:
                        valid_in_chunk = torch.clamp(expert_num_tokens,
                                                     min=0,
                                                     max=CHUNK_SIZE)
                        expert_num_tokens -= CHUNK_SIZE

                    self.fused_experts.apply(
                        fused_out[begin_out_idx:end_out_idx],
                        curr_a1q,
                        w1,
                        w2,
                        curr_topk_weights,
                        curr_topk_ids,
                        activation=activation,
                        global_num_experts=global_num_experts,
                        expert_map=expert_map,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        w1_zp=w1_zp,
                        w2_zp=w2_zp,
                        a1q_scale=curr_a1q_scale,
                        a2_scale=curr_a2_scale,
                        workspace13=workspace13,
                        workspace2=workspace2,
                        expert_num_tokens=valid_in_chunk,
                        apply_router_weight_on_input=
                        apply_router_weight_on_input,
                        a1q_scale_rec=a1_scale_rec,  # vllm_gcu for w4a8
                        a2_scale_rec=a2_scale_rec,  # vllm_gcu for w4a8
                    )
        # NOTE: a1 and a1q might be same buffer with output if inplace
        if shared_output is not None:
            output = a1.copy_(shared_output) if inplace else shared_output
        else:
            output = a1.fill_(0) if inplace else torch.zeros_like(a1)
        del a1, a1q

        self.prepare_finalize.finalize(output, fused_out, topk_weights,
                                       topk_ids, apply_router_weight_on_input)

        return output
