#!/usr/bin/env python
# coding=utf-8
from functools import wraps
from contextlib import contextmanager

import torch
import importlib
from packaging import version
from packaging.version import Version
from vllm.utils import round_up
import vllm_gcu.envs as gcu_envs
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context, get_forward_context

try:
    from vllm.utils import is_torch_equal_or_newer
except Exception:

    def is_torch_equal_or_newer(target: str) -> bool:
        """Check if the installed torch version is >= the target version.

        Args:
            target: a version string, like "2.6.0".

        Returns:
            Whether the condition meets.
        """
        try:
            torch_version = version.parse(str(torch.__version__))
            return torch_version >= version.parse(target)
        except Exception:
            # Fallback to PKG-INFO to load the package info, needed by the doc gen.
            return Version(importlib.metadata.version("torch")) >= Version(target)


def dump_memory_snapshot_when_exception(name):
    def inner(func):
        import vllm_gcu.envs as gcu_envs

        n = gcu_envs.VLLM_DUMP_SNAPSHOT_EVERY_N_STEP
        if n <= 0:
            return func

        torch.gcu.memory._record_memory_history()
        step = 0

        @wraps(func)
        def _wrapper(*args, **kwargs):
            nonlocal step
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            if step % n == 0:
                filename = f"/tmp/vllm_snapshot_rank{rank}_{name}{step}.pkl"
                torch.gcu.memory._dump_snapshot(filename)
            step += 1
            try:
                return func(*args, **kwargs)
            except Exception as err:
                filename = f"/tmp/vllm_snapshot_rank{rank}_exception.pkl"
                torch.gcu.memory._dump_snapshot(filename)
                raise err

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


def ep_alltoall_threshold(vllm_config: VllmConfig):
    """
    Use dynamic memory allocation in EP dispatch when num_tokens_across_dp > threshold,
    use static allocation otherwise. Cudagraph only supports staitc shape,
    so we must ensure threshold >= max_capture_size * dp_size. Decode prefers static.
    """
    threshold = max(
        vllm_config.scheduler_config.max_num_seqs,
        vllm_config.compilation_config.max_capture_size,
    )
    # if vllm_config.speculative_config is not None:
    #     threshold *= vllm_config.speculative_config.num_speculative_tokens + 1
    if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
        sp_size = vllm_config.parallel_config.tensor_parallel_size
        threshold = round_up(threshold, sp_size)
    threshold *= vllm_config.parallel_config.data_parallel_size

    return threshold


@contextmanager
def set_gcu_forward_context(
    attn_metadata,
    vllm_config,
    virtual_engine=0,
    num_tokens=None,
    num_tokens_across_dp=None,
    skip_cuda_graphs=False,
):
    with set_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine,
        num_tokens,
        num_tokens_across_dp,
        skip_cuda_graphs,
    ) as ctx:
        forward_context = get_forward_context()
        threshold = ep_alltoall_threshold(vllm_config)
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is not None:
            total_tokens = dp_metadata.cu_tokens_across_dp_cpu[-1].item()
        else:
            if attn_metadata is not None and hasattr(attn_metadata,
                                                 "num_prefill_tokens"):
                # for v0 attention backends
                total_tokens = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
                if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
                    sp_size = vllm_config.parallel_config.tensor_parallel_size
                    total_tokens = round_up(total_tokens, sp_size)
            else:
                # for v1 attention backends or no attn_metadata
                total_tokens = num_tokens or 0
        use_all2all_v = total_tokens <= threshold
        forward_context.skip_cuda_graphs |= not use_all2all_v
        setattr(forward_context, "all2allv_threshold", None if not use_all2all_v else threshold)

        try:
            yield ctx
        finally:
            if hasattr(forward_context, "all2allv_threshold"):
                delattr(forward_context, "all2allv_threshold")
