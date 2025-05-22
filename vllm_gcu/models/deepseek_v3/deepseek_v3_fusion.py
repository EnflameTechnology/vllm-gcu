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
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, Iterable, Mapping, Optional, Set, Tuple, Union
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, set_current_vllm_config, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)

from vllm.forward_context import get_forward_context
from vllm.inputs import DummyData, INPUT_REGISTRY, InputContext
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    PPMissingLayer,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SequenceData
from vllm_gcu.kernels import _custom_ops as ops

import vllm_gcu.envs as gcu_envs
from vllm_gcu.kernels.fused_moe import fused_experts_impl
from vllm_gcu.kernels.linear import MergedReplicatedLinear
from vllm_gcu.kernels.quantization.fp8 import apply_w8a8_block_fp8_linear


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekFusedQKVProj(MergedReplicatedLinear):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = True, skip_bias_add: bool = False, params_dtype: torch.dtype | None = None, quant_config: QuantizationConfig | None = None, prefix: str = ""):

        super().__init__(input_size, output_sizes, bias, skip_bias_add, params_dtype, quant_config, prefix)
        self.quant_config = quant_config
        if self.quant_config.get_name() in ['awq_gcu', 'moe_wna16_gcu']:
            self.pack_factor = int(self.quant_config.pack_factor)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        if self.quant_config is None:
            assert "DeepseekFusedQKVProj unsupported quant method"
        elif self.quant_config.get_name() in ['awq_gcu', 'moe_wna16_gcu']:
            outs = tuple(torch.empty(x.shape[:-1]+(i,),
                                     dtype=x.dtype, device=x.device)
                         for i in self.output_sizes)
            if x.numel() == 0:
                return outs

            qweight = self.qweight
            scales = self.scales
            qzeros = self.qzeros
            torch.ops._C.fused_qkv_gemm_quant(outs[0], outs[1], x, qweight, scales,
                                              qzeros, self.quant_config.group_size)
            return outs
        elif self.quant_config.get_name() in ['fp8_gcu']:
            assert self.quant_config.weight_block_size is not None
            outs = tuple(torch.empty(x.shape[:-1]+(i,),
                                     dtype=x.dtype, device=x.device)
                         for i in self.output_sizes)
            if x.numel() == 0:
                return outs
            assert self.quant_config.weight_block_size[0] == self.quant_config.weight_block_size[1]
            assert self.input_scale is None

            q_input, x_scale = ops.per_token_group_quant_fp8(
                x,
                self.quant_config.weight_block_size[1],
                dtype=torch.float8_e4m3fn,
                column_major_scales=False,
            )
            torch.ops._C.fused_qkv_proj(outs[0], outs[1], q_input, self.weight,
                                        x_scale, self.weight_scale_inv,
                                        self.quant_config.weight_block_size[0])

            return outs
        else:
            assert "DeepseekFusedQKVProj unsupported quant method"


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
            self.qkv_a_proj_with_mqa = DeepseekFusedQKVProj(
                hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_a_proj_with_mqa",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
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

        self.q_proj_outside = (
            True if quant_config.get_name().startswith("fp8") else False
        )

        if self.q_proj_outside:
            q_proj = nn.Identity()
        else:
            q_proj = self.q_proj if self.q_lora_rank is None else self.q_b_proj

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
            rotary_emb=self.rotary_emb,
            q_proj=q_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            kv_a_layernorm=self.kv_a_layernorm,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            ckq, kv_c_and_k_pe = self.qkv_a_proj_with_mqa(hidden_states)
            hidden_states_or_q_c = self.q_a_layernorm(ckq)
        else:
            hidden_states_or_q_c = hidden_states
            kv_c_and_k_pe = self.kv_a_proj_with_mqa(hidden_states)[0]

        if self.q_proj_outside:
            q_proj = self.q_proj if self.q_lora_rank is None else self.q_b_proj
            hidden_states_or_q_c = q_proj(hidden_states_or_q_c)[0].unsqueeze(0)

        return self.mla_attn(
            hidden_states_or_q_c,
            kv_c_and_k_pe,
            kv_c_and_k_pe,  # place holder
            output_shape=hidden_states.shape
        )
