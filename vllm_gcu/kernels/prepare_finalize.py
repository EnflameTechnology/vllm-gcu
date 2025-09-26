from typing import Optional
from contextlib import nullcontext

import torch
import torch_gcu

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEActivationFormat, FusedMoEPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.distributed import get_ep_group
from vllm.platforms import current_platform
from vllm.forward_context import ForwardContext, get_forward_context

from vllm_gcu.distributed.parallel_state import all_to_all_v2
import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels import _custom_ops as ops


class MoEPrepareAndFinalizeNoEP(FusedMoEPrepareAndFinalize):

    def __init__(self):
        super().__init__()
        self.shared_experts = None
        self.routed_scaling_factor = 1.0

    def set_shared_experts(self, shared_experts, routed_scaling_factor):
        if shared_experts is not None:
            assert routed_scaling_factor is not None
        self.shared_experts = shared_experts
        self.routed_scaling_factor = routed_scaling_factor

    @property
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert not apply_router_weight_on_input

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1, a1_scale, quant_config.quant_dtype,
            quant_config.per_act_token_quant, quant_config.block_shape)

        total_tokens = torch.full(
            (1, ),
            a1.shape[0],
            dtype=torch.int32,
            device=a1.device,
        )
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(a1)

        return a1q, a1q_scale, total_tokens, topk_ids, topk_weights, shared_output

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        fused_expert_output.mul_(self.routed_scaling_factor)
        output.add_(fused_expert_output)


class AlltoAllPrepareAndFinalize(FusedMoEPrepareAndFinalize):
    """
    AlltoAll impl for the [Quantize-Prepare] and [Finalize] steps.
    """

    def __init__(
        self,
        num_dispatchers: int,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.shared_experts = None
        self.routed_scaling_factor = 1.0
        self.ep_group = get_ep_group().device_group

    @property
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def set_shared_experts(self, shared_experts, routed_scaling_factor):
        if shared_experts is not None:
            assert routed_scaling_factor is not None
        self.shared_experts = shared_experts
        self.routed_scaling_factor = routed_scaling_factor

    def route(self, num_experts, M, topk_ids, log2phy):
        ep_size = self.ep_group.size()
        expert_per_rank = num_experts // ep_size
        if M == 0:
            ep_split_size = torch.zeros([ep_size],
                                        dtype=torch.int32,
                                        device=topk_ids.device)
            ep_token_indices = torch.empty(
                [M * topk_ids.shape[1]],
                dtype=torch.int32,
                device=topk_ids.device,
            )
            send_token_total = torch.zeros([1],
                                           dtype=torch.int32,
                                           device=topk_ids.device)
        else:
            ep_split_size = torch.empty([ep_size],
                                        dtype=torch.int32,
                                        device=topk_ids.device)
            ep_token_indices = torch.empty(
                [M * topk_ids.shape[1]],
                dtype=torch.int32,
                device=topk_ids.device,
            )
            send_token_total = torch.empty([1],
                                           dtype=torch.int32,
                                           device=topk_ids.device)

            if log2phy is not None:
                mapped_topk_ids = log2phy[topk_ids]
            else:
                mapped_topk_ids = topk_ids

            ops.get_ep_indices(
                ep_split_size,
                ep_token_indices,
                send_token_total,
                mapped_topk_ids,
                expert_per_rank,
                ep_size,
            )
        return ep_split_size, ep_token_indices, send_token_total

    def pack(self, tensors, dtype, ep_token_indices):
        assert all([
            i.shape[1] * i.element_size() % dtype.itemsize == 0
            for i in tensors
        ])
        width_as_dtype = [
            i.shape[1] * i.element_size() // dtype.itemsize for i in tensors
        ]
        origin_dtypes = [i.dtype for i in tensors]

        send_packed = torch.cat(
            [i.view(dtype) for i in tensors],
            dim=1,
        )
        if send_packed.numel() == 0:
            send_packed_sorted = torch.empty(
                [1, send_packed.shape[-1]],
                dtype=send_packed.dtype,
                device=send_packed.device,
            )
        else:
            send_packed_sorted = torch_gcu.gcu.efficient.gcu_index(
                send_packed, [ep_token_indices])
            # send_packed_sorted = send_packed[ep_token_indices.to(torch.int64)]
        return send_packed_sorted, width_as_dtype, origin_dtypes

    def unpack(self, recv_packed, width_as_dtype, dtypes, sp_split_size,
               recv_token_total):
        buffers = [
            torch.empty(
                (recv_packed.shape[0], width),
                dtype=recv_packed.dtype,
                device=recv_packed.device,
            ) for width in width_as_dtype
        ]

        if gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
            torch.ops._C.fused_dispatch_decode(
                buffers,
                recv_packed,
                sp_split_size,
                width_as_dtype,
            )
        else:
            torch.ops._C.dynamic_split(
                buffers,
                recv_packed,
                recv_token_total,
                width_as_dtype,
                1,
            )
        for i in range(len(buffers)):
            buffers[i] = buffers[i].view(dtypes[i])
        return buffers


class AlltoAllStaticShape(AlltoAllPrepareAndFinalize):
    """
    AlltoAll impl for the [Quantize-Prepare] and [Finalize] steps.
    This impl use static shape, mainly for decode.
    """

    def __init__(self, threshold, num_dispatchers: int):
        super().__init__(num_dispatchers)
        self.threshold = threshold

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform any quantization (and/or) dispatching needed
        for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - a1_scale: Optional scales for a1
        - a2_scale: Optional scales for the second MoE gemm.  Required to make
          sure the quantization is consistent for both gemms.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.

        Returns a tuple of:
        - quantized + dispatched a.
        - quantized + dispatched a1_scales.
        - Optional tensor as big as number of local experts that contains the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert not apply_router_weight_on_input
        log2phy = None
        hidden_states = a1
        hidden_states_ori = hidden_states
        do_quant = quant_config.is_quantized

        # In official vllm master branch, "per_channel_quant" parameter is used to determine dynamic or static quant
        # TODO: when upgrade, need to refine this parameter
        # https://github.com/vllm-project/vllm/blob/v0.9.0.1/vllm/model_executor/layers/fused_moe/fused_moe.py#L1222
        input_static_quant = (a1_scale is not None)
        all_to_all_with_scales = (do_quant and not input_static_quant)
        if do_quant:
            hidden_states, a1_scale = moe_kernel_quantize_input(
                A=hidden_states,
                A_scale=a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape)

        ep_split_size, ep_token_indices, send_token_total \
            = self.route(num_experts, hidden_states.shape[0], topk_ids, log2phy)

        enable_parallel_compute = gcu_envs.VLLM_GCU_ENABLE_PARALLEL_COMPUTE
        parallel_compute_context = (torch.gcu.ParallelCompute(
            2, 10) if current_platform.is_device_capability(130)
                                    and enable_parallel_compute else
                                    nullcontext())

        send_packed_sorted, transfer_width_as_dtype, origin_dtypes = self.pack(
            [hidden_states, topk_ids, topk_weights] +
            ([a1_scale] if all_to_all_with_scales else []),
            hidden_states.dtype,
            ep_token_indices,
        )
        recv_packed = torch.empty(
            (
                self.threshold,
                sum(transfer_width_as_dtype),
            ),
            dtype=send_packed_sorted.dtype,
            device=send_packed_sorted.device,
        )

        sp_split_size = torch.empty_like(ep_split_size)
        with parallel_compute_context:
            work = all_to_all_v2(
                recv_packed,
                send_packed_sorted,
                sp_split_size,
                ep_split_size,
                group=self.ep_group,
                flag=1,
                async_op=enable_parallel_compute,
            )

            shared_output = None
            if self.shared_experts is not None and hidden_states_ori.shape[
                    0] > 0:
                if a1_scale is not None and not input_static_quant:
                    shared_output = self.shared_experts(
                        hidden_states, a1_scale)
                else:
                    shared_output = self.shared_experts(hidden_states_ori)

            if enable_parallel_compute:
                work.wait()

        recv_token_total = torch.sum(sp_split_size, 0, True, dtype=torch.int32)

        unpacked = self.unpack(recv_packed, transfer_width_as_dtype,
                               origin_dtypes, sp_split_size, recv_token_total)
        hidden_states, topk_ids, topk_weights = unpacked[0:3]
        if all_to_all_with_scales:
            a1_scale = unpacked[3]

        self.ep_token_indices = ep_token_indices
        self.ep_split_size = ep_split_size
        self.sp_split_size = sp_split_size

        return hidden_states, a1_scale, recv_token_total, topk_ids, topk_weights, shared_output

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        """
        # NOTE: we assume output is shared_output or zeros out

        sp_hidden_states = torch.zeros(
            (self.ep_token_indices.shape[0], fused_expert_output.shape[1]),
            dtype=fused_expert_output.dtype,
            device=fused_expert_output.device,
        )

        all_to_all_v2(
            sp_hidden_states,
            fused_expert_output,
            self.ep_split_size,
            self.sp_split_size,
            group=self.ep_group,
            flag=0,
        )

        if output.numel() != 0:
            output.index_add_(
                0,
                self.ep_token_indices,
                sp_hidden_states,
                alpha=self.routed_scaling_factor,
            )

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.threshold // self.num_dispatchers_


class AlltoAllDynamicShape(AlltoAllPrepareAndFinalize):
    """
    AlltoAll impl for the [Quantize-Prepare] and [Finalize] steps.
    This impl use dynamic shape, mainly for prefill.
    """

    def __init__(self, num_dispatchers: int):
        super().__init__(num_dispatchers)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform any quantization (and/or) dispatching needed
        for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - a1_scale: Optional scales for a1
        - a2_scale: Optional scales for the second MoE gemm.  Required to make
          sure the quantization is consistent for both gemms.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.

        Returns a tuple of:
        - quantized + dispatched a.
        - quantized + dispatched a1_scales.
        - Optional tensor as big as number of local experts that contains the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert not apply_router_weight_on_input
        log2phy = None
        hidden_states = a1
        hidden_states_ori = hidden_states
        do_quant = quant_config.is_quantized

        # In official vllm master branch, "per_channel_quant" parameter is used to determine dynamic or static quant
        # TODO: when upgrade, need to refine this parameter
        # https://github.com/vllm-project/vllm/blob/v0.9.0.1/vllm/model_executor/layers/fused_moe/fused_moe.py#L1222
        input_static_quant = (a1_scale is not None)
        all_to_all_with_scales = (do_quant and not input_static_quant)
        if do_quant:
            hidden_states, a1_scale = moe_kernel_quantize_input(
                A=hidden_states,
                A_scale=a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape)

        ep_split_size, ep_token_indices, send_token_total \
            = self.route(num_experts, hidden_states.shape[0], topk_ids, log2phy)

        enable_parallel_compute = gcu_envs.VLLM_GCU_ENABLE_PARALLEL_COMPUTE
        parallel_compute_context = (torch.gcu.ParallelCompute(
            2, 10) if current_platform.is_device_capability(130)
                                    and enable_parallel_compute else
                                    nullcontext())

        sp_split_size = torch.empty_like(ep_split_size)
        torch.distributed.all_to_all_single(sp_split_size,
                                            ep_split_size,
                                            group=self.ep_group)
        recv_token_total = torch.sum(sp_split_size, 0, True, dtype=torch.int32)
        self.cpu_sp_split_size = sp_split_size.cpu().tolist()
        self.cpu_ep_split_size = ep_split_size.cpu().tolist()
        self.cpu_recv_token_total = recv_token_total[0].item()
        self.cpu_send_token_total = send_token_total[0].item()
        ep_token_indices = ep_token_indices[:self.cpu_send_token_total]
        send_packed_sorted, transfer_width_as_dtype, origin_dtypes = self.pack(
            [hidden_states, topk_ids, topk_weights] +
            ([a1_scale] if all_to_all_with_scales else []),
            hidden_states.dtype,
            ep_token_indices,
        )
        recv_packed = torch.empty(
            (
                max(self.cpu_recv_token_total, 1),
                sum(transfer_width_as_dtype),
            ),
            dtype=send_packed_sorted.dtype,
            device=send_packed_sorted.device,
        )
        with parallel_compute_context:
            work = torch.distributed.all_to_all_single(
                recv_packed[:self.cpu_recv_token_total],
                send_packed_sorted[:self.cpu_send_token_total],
                self.cpu_sp_split_size,
                self.cpu_ep_split_size,
                group=self.ep_group,
                async_op=enable_parallel_compute,
            )

            shared_output = None
            if self.shared_experts is not None:
                if a1_scale is not None and not input_static_quant:
                    shared_output = self.shared_experts(
                        hidden_states, a1_scale)
                else:
                    shared_output = self.shared_experts(hidden_states_ori)

            if enable_parallel_compute:
                work.wait()

        unpacked = self.unpack(recv_packed, transfer_width_as_dtype,
                               origin_dtypes, sp_split_size, recv_token_total)
        hidden_states, topk_ids, topk_weights = unpacked[0:3]
        if all_to_all_with_scales:
            a1_scale = unpacked[3]

        self.ep_token_indices = ep_token_indices
        self.ep_split_size = ep_split_size
        self.sp_split_size = sp_split_size

        return hidden_states, a1_scale, recv_token_total, topk_ids, topk_weights, shared_output

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        """
        # NOTE: we assume output is shared_output or zeros out
        sp_hidden_states = torch.zeros(
            (self.ep_token_indices.shape[0], fused_expert_output.shape[1]),
            dtype=fused_expert_output.dtype,
            device=fused_expert_output.device,
        )

        torch.distributed.all_to_all_single(
            sp_hidden_states[:self.cpu_send_token_total],
            fused_expert_output[:self.cpu_recv_token_total],
            self.cpu_ep_split_size,
            self.cpu_sp_split_size,
            group=self.ep_group,
        )

        if output.numel() != 0:
            output.index_add_(
                0,
                self.ep_token_indices,
                sp_hidden_states,
                alpha=self.routed_scaling_factor,
            )

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None


class AlltoAllSelector(AlltoAllPrepareAndFinalize):

    def __init__(self, threshold, num_dispatchers: int):
        super().__init__(num_dispatchers)
        self.dynamic = AlltoAllDynamicShape(num_dispatchers)
        self.static = AlltoAllStaticShape(threshold, num_dispatchers)

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def set_shared_experts(self, *args, **kwargs):
        self.dynamic.set_shared_experts(*args, **kwargs)
        self.static.set_shared_experts(*args, **kwargs)

    def prepare(self, *args, **kwargs):
        forward_context: ForwardContext = get_forward_context()
        all2allv_threshold = forward_context.all2allv_threshold if hasattr(
            forward_context, "all2allv_threshold") else None
        use_all2all_v = all2allv_threshold is not None
        if use_all2all_v:
            self.static.threshold = all2allv_threshold
            return self.static.prepare(*args, **kwargs)
        else:
            return self.dynamic.prepare(*args, **kwargs)

    def finalize(self, *args, **kwargs):
        forward_context: ForwardContext = get_forward_context()
        all2allv_threshold = forward_context.all2allv_threshold if hasattr(
            forward_context, "all2allv_threshold") else None
        use_all2all_v = all2allv_threshold is not None
        if use_all2all_v:
            self.static.threshold = all2allv_threshold
            return self.static.finalize(*args, **kwargs)
        else:
            return self.dynamic.finalize(*args, **kwargs)
