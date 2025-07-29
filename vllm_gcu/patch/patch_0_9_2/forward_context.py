from typing import Any, Optional
import torch
from unittest.mock import patch

from vllm.config import ParallelConfig
from vllm.forward_context import DPMetadata
import vllm_gcu.envs as gcu_envs
from vllm_gcu.utils import round_up

def make(
    parallel_config: ParallelConfig,
    attn_metadata: Any,
    num_tokens: int,
    num_tokens_across_dp: Optional[torch.Tensor] = None
) -> "DPMetadata":

    assert parallel_config.data_parallel_size > 1
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    if attn_metadata is not None and hasattr(attn_metadata,
                                                "num_prefill_tokens"):
        # for v0 attention backends
        batchsize = attn_metadata.num_prefill_tokens + \
            attn_metadata.num_decode_tokens
        if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
            sp_size = parallel_config.tensor_parallel_size
            batchsize = round_up(batchsize, sp_size)
    else:
        # for v1 attention backends or no attn_metadata
        batchsize = num_tokens

    # If num_tokens_across_dp is None, it will be computed by all_reduce
    # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
    assert (num_tokens_across_dp is None
            or num_tokens_across_dp[dp_rank] == batchsize)
    if num_tokens_across_dp is None:
        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            batchsize, dp_size, dp_rank)
    max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
    cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
    return DPMetadata(max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu)

patch("vllm.forward_context.DPMetadata.make", make).start()