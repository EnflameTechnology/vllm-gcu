#!/usr/bin/env python
# coding=utf-8
from functools import wraps

import torch
import importlib
from packaging import version
from packaging.version import Version

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
            return Version(importlib.metadata.version('torch')) >= Version(target)


def dump_memory_snapshot_when_exception(func):
    import vllm_gcu.envs as gcu_envs

    n = gcu_envs.VLLM_DUMP_SNAPSHOT_EVERY_N_STEP
    if n <= 0:
        return func

    torch.gcu.memory._record_memory_history()
    step = 0

    @wraps(func)
    def _wrapper(*args, **kwargs):
        nonlocal step
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if step % n == 0:
            filename = f"/tmp/vllm_snapshot_rank{rank}_step{step}.pkl"
            torch.gcu.memory._dump_snapshot(filename)
        step += 1
        try:
            return func(*args, **kwargs)
        except Exception as err:
            filename = f"/tmp/vllm_snapshot_rank{rank}_exception.pkl"
            torch.gcu.memory._dump_snapshot(filename)
            raise err

    return _wrapper


def is_vllm_equal(target: str) -> bool:
    try:
        import vllm
        vllm_base_version = version.parse(str(vllm.__version__)).base_version
        target_base_version = version.parse(target).base_version
        return vllm_base_version == target_base_version
    except Exception:
        return False
