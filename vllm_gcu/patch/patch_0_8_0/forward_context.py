import time
from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import patch
import torch
import torch.distributed as dist
from vllm.config import VllmConfig
import vllm.forward_context
from vllm.forward_context import DPMetadata

@contextmanager
def patched_set_forward_context(attn_metadata: Any,
                        vllm_config: VllmConfig,
                        virtual_engine: int = 0,
                        num_tokens: int = 0):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    need_to_track_batchsize = vllm.forward_context.track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        vllm.forward_context.forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None

    # On GCU, under the DP (data parallelism) scheme, DeepSeek V3/V1 uses all2all, \
    # so the current all_reduce will be removed to reduce the gap between decode steps
    # add vllm_config.model_config.architectures[0] != 'DeepseekV3ForCausalLM' based on original codes
    if vllm_config.parallel_config.data_parallel_size > 1 and \
        vllm_config.model_config.architectures[0] != 'DeepseekV3ForCausalLM':
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        if attn_metadata is not None:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
        else:
            batchsize = num_tokens
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = batchsize
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_tensor, dim=0)
        dp_metadata = DPMetadata(cu_tokens_across_dp_cpu)

    prev_context = vllm.forward_context._forward_context
    vllm.forward_context._forward_context = vllm.forward_context.ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata)
    try:
        yield
    finally:
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            torch.cuda.synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            vllm.forward_context.batchsize_forward_time[batchsize].append(
                (now - vllm.forward_context.forward_start_time) * 1000)
            if now - vllm.forward_context.last_logging_time > vllm.forward_context.batchsize_logging_interval:
                vllm.forward_context.last_logging_time = now
                forward_stats = []
                for bs, times in vllm.forward_context.batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    vllm.forward_context.logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)
        vllm.forward_context._forward_context = prev_context

patch("vllm.forward_context.set_forward_context", patched_set_forward_context).start()
patch("vllm_gcu.worker.model_runner.set_forward_context", patched_set_forward_context).start()
