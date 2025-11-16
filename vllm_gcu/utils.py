#!/usr/bin/env python
# coding=utf-8
from typing import Optional
import sys
from functools import wraps, lru_cache
from contextlib import contextmanager

import torch
from packaging import version
from vllm.config import VllmConfig, CUDAGraphMode
import vllm.envs as envs
from vllm.forward_context import set_forward_context, get_forward_context, BatchDescriptor
from vllm.v1.worker.ubatch_utils import UBatchSlices
import vllm_gcu.envs as gcu_envs


def dump_memory_snapshot_when_exception(name):

    def inner(func):
        n = gcu_envs.VLLM_DUMP_SNAPSHOT_EVERY_N_STEP
        if n <= 0:
            return func

        torch.gcu.memory._record_memory_history()
        step = 0

        @wraps(func)
        def _wrapper(*args, **kwargs):
            nonlocal step
            rank = (torch.distributed.get_rank()
                    if torch.distributed.is_initialized() else 0)
            try:
                r = func(*args, **kwargs)
            except Exception as err:
                filename = f"/tmp/vllm_snapshot_rank{rank}_exception.pkl"
                torch.gcu.memory._dump_snapshot(filename)
                raise err
            if step % n == 0:
                filename = f"/tmp/vllm_snapshot_rank{rank}_{name}{step}.pkl"
                torch.gcu.memory._dump_snapshot(filename)
            step += 1
            return r

        return _wrapper

    return inner


def is_vllm_equal(target: str) -> bool:
    try:
        import vllm

        vllm_base_version = version.parse(str(vllm.__version__)).base_version
        target_base_version = version.parse(target).base_version
        return vllm_base_version == target_base_version
    except Exception:
        return False


def is_nixl_equal(target: str):
    try:
        from importlib import metadata
        version = metadata.version('nixl')
        return version == target
    except:
        return False


@lru_cache(maxsize=1)
def _ep_alltoall_threshold(data_parallel_size,
                           max_num_seqs,
                           max_capture_size,
                           num_speculative_tokens=None):
    """
    Use dynamic memory allocation in EP dispatch when num_tokens_across_dp > threshold,
    use static allocation otherwise. Cudagraph only supports staitc shape,
    so we must ensure threshold >= max_capture_size * dp_size. Decode prefers static.
    """
    threshold = max_num_seqs

    if num_speculative_tokens is not None:
        threshold *= num_speculative_tokens + 1

    threshold = max(threshold, max_capture_size)
    threshold *= data_parallel_size

    return threshold


@lru_cache(maxsize=8)
def get_hooks(group: str):
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    return entry_points(group=group)


@contextmanager
def set_gcu_forward_context(
    attn_metadata,
    vllm_config: VllmConfig,
    virtual_engine=0,
    num_tokens=None,
    num_tokens_across_dp=None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: Optional[BatchDescriptor] = None,
    ubatch_slices: Optional[UBatchSlices] = None,
    is_dummy=False,
):
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size > 1 and gcu_envs.VLLM_GCU_SKIP_ACROSS_DP and num_tokens_across_dp is None and num_tokens is not None:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if envs.VLLM_ALL2ALL_BACKEND == 'deepep_low_latency':
            sp_size = vllm_config.parallel_config.tensor_parallel_size \
                if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL or vllm_config.parallel_config.use_sequence_parallel_moe \
        else 1
            if max_num_batched_tokens <= envs.VLLM_MOE_DP_CHUNK_SIZE * sp_size:
                num_tokens = max_num_batched_tokens
                num_tokens_across_dp = torch.full((dp_size, ), num_tokens)
        else:
            if max_num_batched_tokens * dp_size <= _ep_alltoall_threshold(
                    dp_size, vllm_config.scheduler_config.max_num_seqs,
                    vllm_config.compilation_config.max_capture_size,
                    vllm_config.speculative_config.num_speculative_tokens
                    if vllm_config.speculative_config else None):
                num_tokens = max_num_batched_tokens
                num_tokens_across_dp = torch.full((dp_size, ), num_tokens)

    with set_forward_context(
            attn_metadata,
            vllm_config,
            virtual_engine,
            num_tokens,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            batch_descriptor,
            ubatch_slices,
    ) as ctx:
        # invoke hooks
        discovered_hooks = get_hooks(group="vllm_gcu.hooks")
        if len(discovered_hooks) > 0:
            for hook in discovered_hooks:
                func = hook.load()
                func(attn_metadata, vllm_config, num_tokens,
                     num_tokens_across_dp, is_dummy)

        forward_context = get_forward_context()
        threshold = _ep_alltoall_threshold(
            vllm_config.parallel_config.data_parallel_size,
            vllm_config.scheduler_config.max_num_seqs,
            vllm_config.compilation_config.max_capture_size,
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config else None)
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is not None:
            total_tokens = torch.sum(
                dp_metadata.num_tokens_across_dp_cpu).item()
        else:
            if attn_metadata is not None and hasattr(attn_metadata,
                                                     "num_prefill_tokens"):
                # for v0 attention backends
                total_tokens = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends or no attn_metadata
                total_tokens = num_tokens or 0
        use_all2all_v = total_tokens <= threshold

        if not use_all2all_v and envs.VLLM_ALL2ALL_BACKEND not in [
                "deepep_high_throughput", "deepep_low_latency"
        ]:
            forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        setattr(forward_context, "all2allv_threshold",
                None if not use_all2all_v else threshold)

        try:
            yield ctx
        finally:
            if hasattr(forward_context, "all2allv_threshold"):
                delattr(forward_context, "all2allv_threshold")


def prepare_communication_buffer_for_model_noep(
        model: torch.nn.Module) -> None:
    """
    Prepare the communication buffer for the model.
    """

    moe_modules = [
        module for module in model.modules()
        if (module.__class__.__name__ == "FusedMoE"
            or module.__class__.__name__ == "SharedFusedMoE")
    ]
    for module in moe_modules:
        module.quant_method.init_prepare_finalize(module)


def scatter(seqlen, size):
    indices = list(range(size))
    return [(seqlen + indices[i]) // size - indices[i] // size
            for i in range(size)]
