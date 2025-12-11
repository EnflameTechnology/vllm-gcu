# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import numpy as np
from unittest.mock import patch
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import contextlib
from contextlib import AbstractContextManager
from vllm.utils import _current_stream_tls

import torch
from torch.autograd.profiler import record_function
import vllm.envs as envs
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm_gcu.kernels._custom_ops import get_token_bin_counts_and_mask

def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: Optional[int] = 1,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name,
                                       num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            if current_platform.is_cuda() or current_platform.is_xpu() or \
                    current_platform.is_cuda_alike():
                # We know that the GPU runner is not impacted by this
                # case. Some test code depends on runner_kv_caches, but
                # not in a way that's impacted by ignoring this.
                pass
            else:
                raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]

_PROFILER_FUNC = None

def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    global _PROFILER_FUNC

    # fast path assume it is set
    if _PROFILER_FUNC is not None:
        return _PROFILER_FUNC(name)

    func = contextlib.nullcontext
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        func = record_function
    elif envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        pass

    _PROFILER_FUNC = func
    return func(name)

origin_set_stream = torch.gcu.set_stream

def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    _current_stream_tls.value = stream
    origin_set_stream(stream)

torch.cuda.set_stream = _patched_set_stream
torch.gcu.set_stream = _patched_set_stream

patch("vllm.v1.worker.gpu_model_runner.bind_kv_cache", bind_kv_cache).start()
patch("vllm.v1.worker.gpu_model_runner.record_function_or_nullcontext", record_function_or_nullcontext).start()
patch("vllm_gcu.worker.gcu_model_runner.record_function_or_nullcontext", record_function_or_nullcontext).start()
patch("vllm.model_executor.layers.utils.get_token_bin_counts_and_mask", get_token_bin_counts_and_mask).start()
