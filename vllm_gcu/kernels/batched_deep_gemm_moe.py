from typing import Optional
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts, _resize_cache, fp8_m_grouped_gemm_nt_masked,
    silu_mul_fp8_quant_deep_gemm_cuda)


class BatchedDeepGemmExpertsGCU(BatchedDeepGemmExperts):

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
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        assert self.block_shape is not None

        a1q = hidden_states
        _, N, K = w1.size()

        assert w2.size(1) == K

        E, max_num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        # (from deepgemm docs) : A value hint (which is a value on CPU)
        # for the M expectation of each batch, correctly setting this value
        # may lead to better performance.
        expected_m = max_num_tokens * top_k_num // global_num_experts
        fp8_m_grouped_gemm_nt_masked((a1q, a1q_scale), (w1, self.w1_scale),
                                     workspace1, expert_num_tokens, expected_m)

        a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm_cuda(
            workspace1, expert_num_tokens)

        fp8_m_grouped_gemm_nt_masked((a2q, a2q_scale), (w2, self.w2_scale),
                                     output, expert_num_tokens, expected_m)
