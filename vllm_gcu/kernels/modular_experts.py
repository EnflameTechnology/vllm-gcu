from math import prod
from typing import Optional
import torch


from vllm.forward_context import get_forward_context
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_dtype_str
from vllm_gcu.kernels.fused_moe import get_default_config, invoke_fused_moe_kernel, moe_align_block_size


def _resize_cache_with_dtype(x: torch.Tensor, v: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """

    assert prod(v) * dtype.itemsize <= x.numel() * x.dtype.itemsize
    assert dtype.itemsize % x.dtype.itemsize == 0 or x.dtype.itemsize % dtype.itemsize == 0
    return x.flatten()[:prod(v)* dtype.itemsize // x.dtype.itemsize].view(dtype).view(*v)


class TritonExpertsPad(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        per_act_token_quant: bool = False,
        block_shape: Optional[list[int]] = None,
        per_channel_quant: bool = False
    ):
        super().__init__(
            FusedMoEQuantConfig.make(
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
            ))

        self.use_fp8_w8a8 = use_fp8_w8a8
        self.use_int4_w4a16 = use_int4_w4a16
        self.use_int8_w8a8 = use_int8_w8a8
        self.use_int8_w8a16 = use_int8_w8a16

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        workspace1 = (M, topk, max(N // 2, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_num_tokens: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        a1q_scale_rec: Optional[torch.Tensor],
        a2_scale_rec: Optional[torch.Tensor],
    ):
        # TODO:bias
        # Check constraints.
        if self.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), (
                "Hidden size mismatch")
        else:
            if self.use_fp8_w8a8 and w1.dtype == torch.int8:
                # for w4a8-fp8
                assert hidden_states.size(-1) // 2 == w1.size(2), (
                    "Hidden size mismatch")
            else:
                assert hidden_states.size(-1) == w1.size(2), \
                    (f"Hidden size mismatch {hidden_states.size(-1)} "
                        f"!= {w1.size(2)}")

        assert hidden_states.is_contiguous(
        ), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2

        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn,
            torch.int8
        ]

        E, num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        if global_num_experts == -1:
            global_num_experts = E

        config_dtype = get_config_dtype_str(use_fp8_w8a8=self.use_fp8_w8a8,
                                            use_int8_w8a16=self.use_int8_w8a16,
                                            use_int4_w4a16=self.use_int4_w4a16,
                                            dtype=hidden_states.dtype)

        config = get_default_config(
            M=num_tokens,
            E=w2.shape[0],
            N=w2.shape[2],
            K=w1.shape[2],
            topk=top_k_num,
            dtype=config_dtype,
            is_marlin=False,
            block_shape=self.block_shape,
        )

        forward_context = get_forward_context()
        all2allv_threshold = forward_context.all2allv_threshold if hasattr(
            forward_context, "all2allv_threshold") else None
        is_static = all2allv_threshold is not None

        # We can reuse the memory between these because by the time we need
        # cache3, we're done with cache1
        intermediate_cache1 = _resize_cache(workspace2,
                                            (num_tokens, top_k_num, N))
        intermediate_cache2_dtype = self.quant_dtype if self.quant_dtype is not None else hidden_states.dtype
        intermediate_cache2 = _resize_cache_with_dtype(workspace13,
                                            (num_tokens * top_k_num, N // 2),
                                            intermediate_cache2_dtype)
        intermediate_cache3 = _resize_cache(workspace2,
                                            (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'],
                                 global_num_experts, expert_map,
                                 expert_num_tokens))

        invoke_fused_moe_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            topk_weights,  # TODO: remove
            topk_ids,  # TODO: remove
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            use_fp8_w8a8=self.use_fp8_w8a8,
            use_int8_w8a8=self.use_int8_w8a8,
            use_int8_w8a16=self.use_int8_w8a16,
            use_int4_w4a16=self.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            real_token_num=expert_num_tokens if is_static else None,
            A_scale_rec=a1q_scale_rec,
        )

        if activation == "silu":
            if self.use_fp8_w8a8:
                if w1.dtype == torch.int8:
                    # shape = (
                    #     *intermediate_cache2.shape[:-1],
                    #     N // 2 // self.quant_config.block_shape[1],
                    # )
                    # a2_scale = torch.empty(
                    #     shape, dtype=torch.float32, device=intermediate_cache1.device
                    # )
                    assert a2_scale is not None
                    torch.ops._C.silu_mul_static_fp8_quant(
                        intermediate_cache2.view(-1, top_k_num, N // 2),
                        intermediate_cache1,
                        a2_scale,
                        expert_num_tokens,
                    )
                else:
                    group_size = self.quant_config.block_shape[1]
                    shape = (
                        *intermediate_cache2.shape[:-1],
                        N // 2 // group_size,
                    )
                    a2_scale = torch.empty(
                        shape, dtype=torch.float32, device=intermediate_cache1.device
                    )
                    torch.ops._C.silu_mul_per_token_group_quant_with_size(
                        intermediate_cache2.view(-1, top_k_num, N // 2),
                        a2_scale.view(-1, top_k_num, N // 2 // group_size),
                        intermediate_cache1,
                        expert_num_tokens,
                        group_size,
                    )
            elif self.use_int8_w8a8:
                intermediate_cache2_temp = torch.empty_like(intermediate_cache2, dtype=intermediate_cache1.dtype)
                if a2_scale is not None:
                    torch.ops._C.silu_and_mul_pad(
                        intermediate_cache2_temp.view(-1, top_k_num, N // 2),
                        intermediate_cache1,
                        expert_num_tokens,
                    )
                    intermediate_cache2 = intermediate_cache2_temp
                else:
                    torch.ops._C.silu_and_mul_pad(
                        intermediate_cache2_temp.view(-1, top_k_num, N // 2),
                        intermediate_cache1,
                        expert_num_tokens,
                    )
                    a2_scale = torch.empty((intermediate_cache2_temp.numel() // intermediate_cache2_temp.shape[-1], 1), dtype=torch.float32, device="gcu")
                    torch.ops._C.dynamic_scaled_int8_quant(intermediate_cache2, intermediate_cache2_temp, a2_scale, None)
                del intermediate_cache2_temp
            else:
                torch.ops._C.silu_and_mul_pad(
                    intermediate_cache2.view(-1, top_k_num, N // 2),
                    intermediate_cache1,
                    expert_num_tokens,
                )
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            a2_scale,
            w2_scale,
            w2_zp,
            topk_weights,
            topk_ids,  # TODO: remove
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            use_fp8_w8a8=self.use_fp8_w8a8,
            use_int8_w8a8=self.use_int8_w8a8,
            use_int8_w8a16=self.use_int8_w8a16,
            use_int4_w4a16=self.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            real_token_num=expert_num_tokens if is_static else None,
            A_scale_rec=a2_scale_rec,
        )

        torch.ops._moe_C.moe_sum_pad(
            output,
            intermediate_cache3.view(*intermediate_cache3.shape),
            expert_num_tokens,
            1,
            False,
        )
