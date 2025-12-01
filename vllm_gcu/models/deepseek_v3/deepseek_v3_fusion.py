# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import Attention
from vllm.config import CacheConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_gcu.kernels import _custom_ops as ops

import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels.linear import MergedReplicatedLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm_gcu.distributed.sp import sp_to_tp


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2MLAAttentionFusion(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.qkv_fuse = False
            if quant_config is not None:
                # HACK: use q_b_proj as layer since get_quant_method only check it's type
                q_a_method = quant_config.get_quant_method(self.q_b_proj, f"{prefix}.q_a_proj")
                kv_a_method = quant_config.get_quant_method(self.q_b_proj, f"{prefix}.kv_a_proj_with_mqa")
                if type(q_a_method) is type(kv_a_method) and not isinstance(q_a_method, UnquantizedLinearMethod):
                    # UnquantizedLinearMethod: skip is not safe for merge linear
                    self.qkv_fuse = True

            if self.qkv_fuse:
                self.qkv_a_proj_with_mqa = MergedReplicatedLinear(
                    hidden_size,
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.qkv_a_proj_with_mqa",
                )
            else:
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.q_a_proj",
                )
                self.kv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.kv_a_proj_with_mqa",
                )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=not gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale


        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            kv_a_layernorm=self.kv_a_layernorm,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        actual_seqlen = None,
    ) -> torch.Tensor:
        
        if self.q_lora_rank is not None:
            if self.qkv_fuse:
                q_c_and_latent = self.qkv_a_proj_with_mqa(hidden_states)[0]
                if actual_seqlen is not None:
                    q_c_and_latent = sp_to_tp(q_c_and_latent, actual_seqlen)
                q_c, latent_cache = q_c_and_latent.split(
                    self.qkv_a_proj_with_mqa.output_sizes, dim=-1)
            else:
                if actual_seqlen is not None:
                    hidden_states = sp_to_tp(hidden_states, actual_seqlen)
                q_c = self.q_a_proj(hidden_states)[0].contiguous()
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0].contiguous()
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            if actual_seqlen is not None:
                hidden_states = sp_to_tp(hidden_states, actual_seqlen)
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)

        attn_out = self.mla_attn(
            q,
            latent_cache,
            positions,  # place holder
            output_shape=(q.shape[0],
                          self.num_local_heads * self.v_head_dim))

        return self.o_proj(attn_out)[0]
