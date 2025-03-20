#!/usr/bin/env python
# coding=utf-8

from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    RotaryEmbedding,
)


def forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from vllm_gcu.kernels import _custom_ops as ops

    if (
        self.cos_sin_cache.device != query.device
        or self.cos_sin_cache.dtype != query.dtype
    ):
        self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    if offsets is not None:
        ops.batched_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
            self.rotary_dim,
            offsets,
        )
    else:
        # TODO(tianyu): remove hard code after op impl
        q_shape = None
        if query.ndim == 3:
            query = query.clone()
            key = key.clone()

            q_shape = query.shape
            k_shape = key.shape
            query = query.reshape(query.shape[0], -1)
            key = key.reshape(key.shape[0], -1)
        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )

        if q_shape:
            query = query.reshape(q_shape)
            key = key.reshape(k_shape)
    return query, key


def deepseek_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from vllm_gcu.kernels import _custom_ops as ops

    self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

    assert offsets is None
    assert query.ndim == 3
    assert key.ndim == 3

    # Expect as a outplace op
    query = query.clone()
    key = key.clone()

    q_shape = query.shape
    k_shape = key.shape

    query = query.reshape(query.shape[0], -1)
    key = key.reshape(key.shape[0], -1)
    ops.rotary_embedding(
        positions,
        query,
        key,
        self.head_size,
        self.cos_sin_cache,
        self.is_neox_style,
    )

    query = query.reshape(q_shape)
    key = key.reshape(k_shape)

    return query, key


RotaryEmbedding.forward_oot = forward_oot
DeepseekScalingRotaryEmbedding.forward = deepseek_oot
