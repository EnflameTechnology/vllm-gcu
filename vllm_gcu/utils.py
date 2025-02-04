#!/usr/bin/env python
# coding=utf-8
import contextlib
import gc
import os
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Generator

import torch
import torch_gcu


@dataclass
class MemorySnapshot:
    """Memory snapshot."""

    torch_peak: int = 0
    gcu_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0
    auto_measure: bool = True

    def __post_init__(self):
        if self.auto_measure:
            self.measure()

    def measure(self):
        self.torch_peak = torch.gcu.memory_stats().get("allocated_bytes.all.peak", 0)

        self.gcu_memory = torch.gcu.mem_get_info()[1] - torch.gcu.mem_get_info()[0]

        self.torch_memory = torch.gcu.memory_reserved()

        self.non_torch_memory = self.gcu_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            gcu_memory=self.gcu_memory - other.gcu_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            auto_measure=False,
        )


@dataclass
class MemoryProfilingResult:
    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: float = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    before_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    after_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot, weights_memory: int
) -> Generator[MemoryProfilingResult, None, None]:
    gc.collect()
    torch.gcu.empty_cache()
    torch.gcu.reset_peak_memory_stats()

    result = MemoryProfilingResult()

    result.before_create = baseline_snapshot
    # the part of memory used for holding the model weights
    result.weights_memory = weights_memory

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.gcu.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp
    result.non_kv_cache_memory = (
        result.non_torch_increase + result.torch_peak_increase + result.weights_memory
    )  # noqa


def dump_memory_snapshot_when_exception(func):
    n = os.environ.get("VLLM_DUMP_SNAPSHOT_EVERY_N_STEP", 0)
    if n <= 0:
        return func

    torch.gcu.memory._record_memory_history()
    step = 0

    @wraps(func)
    def _wrapper(*args, **kwargs):
        nonlocal step
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if step % int(n) == 0:
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
