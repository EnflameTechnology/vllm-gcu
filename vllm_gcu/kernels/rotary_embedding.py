#!/usr/bin/env python
# coding=utf-8

from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    RotaryEmbedding,
    MRotaryEmbedding,
)
from vllm_gcu.kernels import _custom_ops as ops


def forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

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

def m_forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-native implementation equivalent to forward().

    Args:
        positions:
            [num_tokens,] (text only) or
            [3, num_tokens] (T/H/W positions with multimodal inputs)
        query: [num_tokens, num_heads * head_size]
        key: [num_tokens, num_kv_heads * head_size]
    """
    assert positions.ndim == 1 or positions.ndim == 2
    num_tokens = positions.shape[-1]
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        assert self.mrope_section

        cos = torch.cat([
            m[i]
            for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
        ],
                        dim=-1)
        sin = torch.cat([
            m[i]
            for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
        ],
                        dim=-1)
        cos_sin = torch.concat([cos, sin], -1)
        rotary_dim = cos.shape[0]
        positions = torch.arange(rotary_dim, device=query.device, dtype=torch.long)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., :self.rotary_dim]
    query_pass = query[..., self.rotary_dim:]

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)
    key_rot = key[..., :self.rotary_dim]
    key_pass = key[..., self.rotary_dim:]
    query_rot_shape = query_rot.shape
    key_rot_shape = key_rot.shape
    query_rot = query_rot.reshape([query_rot_shape[0], -1])
    key_rot = key_rot.reshape([key_rot_shape[0], -1])
    ops.rotary_embedding(
        positions,
        query_rot,
        key_rot,
        self.head_size,
        cos_sin,
        self.is_neox_style,
    )
    query_rot = query_rot.reshape(query_rot_shape)
    key_rot = key_rot.reshape(key_rot_shape)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key

RotaryEmbedding.forward_oot = forward_oot
DeepseekScalingRotaryEmbedding.forward = deepseek_oot
MRotaryEmbedding.forward = m_forward_oot
