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
import typing
from typing import Any, Optional, Union, Mapping, Tuple
from collections.abc import Iterable, Callable

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig
import vllm.envs as envs
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    get_current_vllm_config,
    ModelConfig,
    set_current_vllm_config,
    VllmConfig,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
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
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    PPMissingLayer,
)
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context

import vllm_gcu.envs as gcu_envs
import vllm_gcu.distributed.parallel_state  # noqa
from vllm_gcu.kernels.linear import MergedReplicatedLinear, CustomMergedColumnParallelLinear
from vllm_gcu.models.deepseek_v3.deepseek_v3_fusion import DeepseekV2MLAAttentionFusion
from vllm_gcu.distributed.sp import slice_tensor_sp, sp_to_tp, tp_to_sp
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadataDs
from vllm_gcu.kernels.sampler import GCUSampler as SamplerDS
from vllm_gcu.kernels.sampler import ParallelTopKTopPSampler
from vllm_gcu.kernels.rejection_sampler import GCURejectionSampler as RejectionSamplerDS
#from vllm.v1.sample.rejection_sampler import RejectionSampler as RejectionSamplerDS
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    from vllm_gcu.compilation.pass_manager import PassManager
    from vllm.compilation.inductor_pass import pass_context

    vllm_config = get_current_vllm_config()
    with pass_context(None):
        PassManager(vllm_config)(graph)
    graph.eliminate_dead_code()
    return graph


def custom_backend(
    graph: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
):
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx

    current_config = config.get_config_copy()
    current_config["post_grad_custom_post_pass"] = custom_pass
    current_config["enable_auto_functionalized_v2"] = False

    return compile_fx(
        graph, example_inputs, config_patches=current_config
    )


class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
            self.gate_up_proj = MergedReplicatedLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.down_proj",
            )
        else:
            column_cls = CustomMergedColumnParallelLinear if prefix.endswith("shared_experts") else MergedColumnParallelLinear
            self.gate_up_proj = column_cls(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.down_proj",
            )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x, x_scale: Optional[torch.Tensor]=None):
        if x_scale is not None:
            # only support fp8
            assert x.dtype in [torch.float8_e4m3fn]
            gate_up, _ = self.gate_up_proj(x, x_scale)
        else:
            gate_up, _ = self.gate_up_proj(x)

        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        model_config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_log2phy=None,
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = (envs.VLLM_ALL2ALL_BACKEND
                                     in ("deepep_high_throughput",
                                         "deepep_low_latency")
                                     and ((parallel_config.enable_expert_parallel
                                     and self.tp_size > 1) or gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL))

        if layer_log2phy is not None:
            self.layer_log2phy = layer_log2phy.to(torch.gcu.current_device())
        else:
            self.layer_log2phy = layer_log2phy

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts if enable_eplb else 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = (self.n_logical_experts +
                                   self.n_redundant_experts)
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        if config.n_shared_experts is None:
            self.experts = FusedMoE(
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.scoring_func,
                # we do scaling outside, set factor to 1.0 to avoid double mul
                routed_scaling_factor=1.0,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )
            self.shared_experts = None
        else:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
            if quant_config is not None:
                self.shared_experts = torch.compile(
                    self.shared_experts,
                    backend=custom_backend,
                    dynamic=True,
                )

            self.experts = SharedFusedMoE(
                shared_experts=self.shared_experts,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )
            # NOTE: just for alltoall, fuse add into index_add,
            # if we only use deepep, adding it externally makes no difference
            self.experts.add_shared = True

        if self.experts.ep_size > 1 and (
                self.experts.dp_size > 1
                or gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL):
            self.tp_size = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)
        fused_moe_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if self.n_shared_experts is not None:
            _, final_hidden_states = fused_moe_out
            shared_output = None
        else:
            final_hidden_states = fused_moe_out
            shared_output = None

        if shared_output is not None:
            final_hidden_states *= self.routed_scaling_factor
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
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
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
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
        # O projection.
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
            self.use_normal_rope = False
        else:
            self.use_normal_rope = True
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

        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        actual_seqlen = None,
    ) -> torch.Tensor:
        assert actual_seqlen is None
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        if self.use_normal_rope:
            seq_len = positions.size(0)
            ori_q_pe_shape, ori_k_pe_shape = q_pe.shape, k_pe.shape
            q_pe = q_pe.reshape(seq_len, -1)
            k_pe = k_pe.reshape(seq_len, -1)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if self.use_normal_rope:
            q_pe, k_pe = q_pe.view(ori_q_pe_shape), k_pe.view(ori_k_pe_shape)

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., : self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV2MLAAttention(nn.Module):
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
        rope_scaling: Optional[dict[str, Any]] = None,
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
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
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
        )

        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        actual_seqlen = None,
    ) -> torch.Tensor:
        if actual_seqlen is not None:
            hidden_states = sp_to_tp(hidden_states, actual_seqlen)
        if self.q_lora_rank is not None:
            q_c = self.q_a_proj(hidden_states)[0]
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_local_heads * self.v_head_dim))
        return self.o_proj(attn_out)[0]


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prior_expert_map: Optional[dict[int, torch.Tensor]] = None,
        log2phy=None,
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        self.layer_idx = int(prefix.split(sep=".")[-1])
        if model_config.use_mla:
            if gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
                attn_cls = DeepseekV2MLAAttentionFusion
            else:
                attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (
            config.n_routed_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % config.moe_layer_freq == 0
        ):
            layer_log2phy = (
                log2phy[self.layer_idx - config.first_k_dense_replace]
                if log2phy is not None
                else None
            )

            self.mlp = DeepseekV2MoE(
                config=config,
                model_config=model_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                layer_log2phy=layer_log2phy,
                enable_eplb=enable_eplb,
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.model_config = model_config
        self.config = config

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        actual_seqlen: Optional[int],
    ) -> torch.Tensor:

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
            assert actual_seqlen is not None
            # add mtp layer
            if self.layer_idx % self.config.num_hidden_layers == 0:
                residual = slice_tensor_sp(residual, actual_seqlen)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            actual_seqlen=actual_seqlen if self.layer_idx %
            self.config.num_hidden_layers != 0 else None,
        )

        if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
            hidden_states = tp_to_sp(hidden_states, actual_seqlen)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

class SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size,
                                   config.hidden_size,
                                   quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.eh_proj"
        )
        self.shared_head = SharedHead(config=config, quant_config=quant_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_block = DeepseekV2DecoderLayer(config, prefix, model_config,
                                                cache_config, quant_config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds[positions == 0] = 0
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))[0]

        actual_seqlen = hidden_states.shape[0]

        hidden_states, residual = self.mtp_block(positions=positions,
                                                 hidden_states=hidden_states,
                                                 residual=None,
                                                 actual_seqlen=actual_seqlen)
        hidden_states = residual + hidden_states

        if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
            hidden_states = sp_to_tp(hidden_states, actual_seqlen)

        return hidden_states

def compile_with_fused_mtp_consider(compile_entry_fused_mtp: bool):
    def cls_decorator_helper(cls):
        if compile_entry_fused_mtp:
            if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
                return support_torch_compile(cls)
        else:
            if not gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
                return support_torch_compile(cls)
        return cls
    return cls_decorator_helper


@compile_with_fused_mtp_consider(False)
class DeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        prior_expert_map: Optional[dict[int, torch.Tensor]] = None,
        log2phy=None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        enable_eplb = vllm_config.parallel_config.enable_eplb
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prior_expert_map=prior_expert_map,
                log2phy=log2phy,
                enable_eplb=enable_eplb,
            ),
            prefix=f"{prefix}.layers",
        )
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 0)
        if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
            mtp_layers = torch.nn.ModuleDict({
                str(idx):
                    DeepSeekMultiTokenPredictorLayer(
                        vllm_config,
                        f"{prefix}.layers.{idx}"
                    )
                for idx in range(config.num_hidden_layers,
                                 config.num_hidden_layers + self.num_mtp_layers)
            })
            self.layers.extend(mtp_layers.values())
            self.mtp_start_layer = self.end_layer
            self.mtp_end_layer = self.mtp_start_layer + self.num_mtp_layers
        else:
            self.mtp_start_layer = self.end_layer
            self.mtp_end_layer = self.mtp_start_layer + self.num_mtp_layers

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward_mtp(
        self,
        input_ids: torch.Tensor,  # bs
        positions: torch.Tensor,  # bs
        previous_hidden_states: torch.Tensor,  # bs x dim
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)

        return self.layers[self.mtp_start_layer + current_step_idx](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None

        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        actual_seqlen = hidden_states.shape[0]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                actual_seqlen,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL:
            hidden_states = sp_to_tp(hidden_states, actual_seqlen)
        return hidden_states

@compile_with_fused_mtp_consider(True)
class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        parallel_config = vllm_config.parallel_config

        additional_config = vllm_config.additional_config

        self.num_speculative_tokens = \
            vllm_config.speculative_config.num_speculative_tokens if \
                vllm_config.speculative_config is not None else 0
        
        self.use_torch_compile = False
        if vllm_config.compilation_config.use_inductor:
            self.use_torch_compile = True
        if vllm_config.additional_config["deepseek_fused_mtp"]:
            self.layer_name = "ds_main_with_mtp"
        else:
            self.layer_name = "ds_main"
        if self.use_torch_compile:
            if self.layer_name in vllm_config.compilation_config.static_forward_context:
                raise ValueError("Duplicate layer name: {}".format(self.layer_name))
            vllm_config.compilation_config.static_forward_context[self.layer_name] = self

        if (
            parallel_config.enable_expert_parallel
            and additional_config
            and "expert_map_path" in additional_config
        ):
            expert_map_path = additional_config["expert_map_path"]
            phy2log, log2phy = torch.load(
                expert_map_path, map_location=torch.device("cpu")
            )

            phy2log = phy2log.to(torch.int32)
            log2phy = log2phy.to(torch.int32)

            num_moe_layers, num_devices, experts_per_device = phy2log.shape
            num_global_experts = num_devices * experts_per_device
            num_dense_layers = vllm_config.model_config.hf_config.first_k_dense_replace
            num_total_layers = num_dense_layers + num_moe_layers

            prior_expert_map = torch.full(
                (num_total_layers, num_devices, num_global_experts),
                -1,
                dtype=torch.int32,
            )
            for layer_idx in range(num_dense_layers, num_total_layers):
                for device_idx in range(num_devices):
                    expert_ids_cur_device = phy2log[
                        layer_idx - num_dense_layers, device_idx
                    ]
                    prior_expert_map[layer_idx, device_idx, expert_ids_cur_device] = (
                        torch.arange(experts_per_device, dtype=torch.int32)
                    )
        else:
            prior_expert_map = None
            log2phy = None

        self.model = DeepseekV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            prior_expert_map=prior_expert_map,
            log2phy=log2phy,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.vllm_config = vllm_config

        self.expert_weights = []

        # Set MoE hyperparameters
        self.num_moe_layers = (config.num_hidden_layers -
                               config.first_k_dense_replace)
        self.num_expert_groups = config.n_group

        self.moe_layers: list[FusedMoE] = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if vllm_config.additional_config["deepseek_fused_mtp"] \
                and isinstance(layer, DeepSeekMultiTokenPredictorLayer):
                continue

            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is not None:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts
        else:
            self.num_redundant_experts = 0

        self.sampler = SamplerDS()
        logprobs_mode = self.sampler.topk_topp_sampler.logprobs_mode
        self.sampler.topk_topp_sampler = ParallelTopKTopPSampler(logprobs_mode)
        self.rejection_sampler = RejectionSamplerDS()
        self.use_dp = vllm_config.parallel_config.data_parallel_size > 1

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        def get_expert_weights(layer):
            weights = list(layer.named_parameters())
            assert all(weight.is_contiguous() for _, weight in weights)

            NON_EXPERT_WEIGHTS = {
                "e_score_correction_bias",
                "w13_input_scale",
                "w2_input_scale",
                "w13_input_scale_rec",
                "w2_input_scale_rec",
            }

            return [
                weight.view(layer.local_num_experts, -1) for name, weight in weights
                if name not in NON_EXPERT_WEIGHTS and weight.shape != torch.Size(
                []) and not name.startswith("_shared_experts.")
            ]

        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(get_expert_weights(layer))
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)
    
    def fused_mtp(self,
                  ds_hidden_states,
                  input_ids,
                  positions,
                  temperature,
                  top_p,
                  top_k,
                  draft_tokens):
        attn_metadata = get_forward_context().attn_metadata
        if ds_hidden_states.shape[0] == 0: 
            #dummy_run(0)
            self.dummy_mtp(ds_hidden_states, input_ids, positions)
        if attn_metadata is None or ds_hidden_states.shape[0] == 0:
            # profile run
            return torch.empty_like(input_ids), \
                   torch.empty((draft_tokens.shape[0],self.num_speculative_tokens+1), dtype = torch.int32, device = draft_tokens.device),\
                   torch.empty((draft_tokens.shape[0]), dtype = torch.int32, device = draft_tokens.device), \
                   torch.empty_like(draft_tokens), \
                   torch.empty((draft_tokens.shape[0], 1), dtype = torch.int32, device = draft_tokens.device)
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]

        if hasattr(attn_metadata,"is_for_decode_gcu_graph") and \
            attn_metadata.is_for_decode_gcu_graph:
            main_model_sampled_tokens, accepted_tokens, accepted_lens, draft_tokens = self.fused_mtp_pure_decode(
                ds_hidden_states, input_ids, positions,
                temperature, top_p, top_k, draft_tokens, attn_metadata)
            main_model_sampled_tokens = main_model_sampled_tokens.squeeze(-1)
        else:
            _, accepted_tokens, accepted_lens, draft_tokens = self.fused_mtp_prefill_decode(
                ds_hidden_states, input_ids, positions,
                temperature, top_p, top_k, draft_tokens, attn_metadata)
            main_model_sampled_tokens = torch.empty_like(input_ids)

        if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION and \
            self.vllm_config.scheduler_config.async_scheduling:
            batch_size = accepted_lens.numel()
            batch_indices = torch.arange(batch_size,
                                        dtype=torch.int,
                                        device=accepted_tokens.device)
            next_token_ids = accepted_tokens[batch_indices, accepted_lens - 1].unsqueeze(1)
        else:
            next_token_ids = torch.empty((draft_tokens.shape[0], 1), dtype = torch.int32, device = draft_tokens.device)

        return main_model_sampled_tokens, accepted_tokens, accepted_lens, draft_tokens, next_token_ids
    
    def dummy_mtp(self, ds_hidden_states, input_ids, positions):
        spec_k = self.num_speculative_tokens
        iter = spec_k if self.use_dp else 1
        for i in range(iter):
            self.model.forward_mtp(
                input_ids = input_ids,
                positions = positions,
                inputs_embeds = None,
                previous_hidden_states = ds_hidden_states,
                spec_step_idx = 0
            )
    
    def fused_mtp_pure_decode(self,
                                 ds_hidden_states,
                                 input_ids,
                                 positions,
                                 temperature,
                                 top_p,
                                 top_k,
                                 draft_tokens,
                                 attn_metadata):
        spec_k = self.num_speculative_tokens
        bsz = ds_hidden_states.shape[0] // (spec_k + 1)
        selected_hidden_states = ds_hidden_states

        # [(num_decodes x (spec_k + 1)) , vocab_size]
        selected_logits = self.compute_logits(selected_hidden_states)
        sampling_metadata = SamplingMetadataDs(
            temperature=temperature,
            all_greedy=False,
            all_random=False,
            top_p=top_p,
            top_k=top_k,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.full_like(temperature, fill_value=0.1),
            presence_penalties=torch.full_like(temperature, fill_value=0.1),
            repetition_penalties=torch.full_like(temperature, fill_value=0.1),
            output_token_ids=[[] for _ in range(bsz)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        sampler_output = self.sampler(logits=selected_logits,
                                                      sampling_metadata=sampling_metadata)
        selected_tokens = sampler_output.sampled_token_ids
        target_token_ids = selected_tokens
        draft_token_ids = draft_tokens
        target_token_ids = target_token_ids.reshape(bsz, spec_k + 1)
        draft_probs = None
        padded_spec_logits = selected_logits
        accepted_tokens, accepted_lens = \
            self.rejection_sampler.rejection_sampler_forward_with_fused_mtp(
                                bsz,
                                spec_k,
                                draft_token_ids,
                                draft_probs,
                                padded_spec_logits,
                                target_token_ids,
                                sampling_metadata
                            )
        mtp_input_ids = accepted_tokens.flatten()  # before [num_decodes x (spec_k + 1) , 1], after [num_decodes x (spec_k + 1)]
        mtp_positions = positions
        mtp_hidden_states = ds_hidden_states
        selected_token_index = (accepted_lens - 1).type(torch.long).unsqueeze(-1)
        for i in range(spec_k):
            mtp_hidden_states = self.model.forward_mtp(
                input_ids=mtp_input_ids,
                positions=mtp_positions,
                inputs_embeds=None,
                previous_hidden_states=mtp_hidden_states,
                spec_step_idx=i,
            )
            mtp_logits = self.compute_logits(mtp_hidden_states, is_mtp_layer=True)

            mtp_token_ids = torch.argmax(mtp_logits, dim=-1, keepdim=True)
            accepted_mtp_tokens = mtp_token_ids.reshape(bsz, spec_k + 1).gather(dim=1,index=selected_token_index)
            draft_tokens[:, i] = accepted_mtp_tokens.squeeze(1)
            # we only support mtp1 for now
            if spec_k == 1:
                break
            mtp_input_ids = mtp_token_ids
        return selected_tokens, accepted_tokens, accepted_lens, draft_tokens

    def fused_mtp_prefill_decode(self,
                                 ds_hidden_states,
                                 input_ids,
                                 positions,
                                 temperature,
                                 top_p,
                                 top_k,
                                 draft_tokens,
                                 attn_metadata):
        device = ds_hidden_states.device
        spec_k = self.num_speculative_tokens
        dtype = ds_hidden_states.dtype
        # adjust for mla（和 attention builder 中呼应）
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decodes
        # [num_decodes + num_prefills + 1]
        query_start_loc = attn_metadata.query_start_loc
        prefill_query_start = num_decodes * (spec_k + 1)

        # Sample next
        # [(num_decodes x (spec_k + 1)) , d_h]
        decode_hidden_states = torch.empty(0, device=device, dtype=dtype)
        prefill_hidden_states = torch.empty(0, device=device, dtype=dtype)  # [num_prefills, d_h]
        if num_decodes > 0:
            # for requests in decoding stage, sample every output hidden_states
            # [(num_decodes x (spec_k + 1)) , d_h]
            decode_hidden_states = ds_hidden_states[:prefill_query_start]
        if num_prefills > 0:
            # for requests in prefilling stage, sample the last output hidden_states
            prefill_hidden_states = torch.index_select(
                ds_hidden_states, dim=0, index=(query_start_loc[num_decodes + 1:] - 1))  # [num_prefills , d_h]
        selected_hidden_states = torch.cat(
            (decode_hidden_states, prefill_hidden_states), dim=0)  # [(num_decodes x (spec_k + 1)) + num_prefills , d_h]

        # [(num_decodes x (spec_k + 1)) + num_prefills , vocab_size]
        selected_logits = self.compute_logits(selected_hidden_states)
        # sample_output: SamplerOutput = self.sample(selected_logits, sampling_metadata)
        # [(num_decodes x (spec_k + 1)) + num_prefills]
        # selected_tokens = sample_output.sampled_token_ids.squeeze(-1)
        sampling_metadata = SamplingMetadataDs(
            temperature=temperature,
            all_greedy=False,
            all_random=False,
            top_p=top_p,
            top_k=top_k,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.full_like(temperature, fill_value=0.1),
            presence_penalties=torch.full_like(temperature, fill_value=0.1),
            repetition_penalties=torch.full_like(temperature, fill_value=0.1),
            output_token_ids=[[] for _ in range(num_prefills + num_decodes)], # not used temporarily
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        sampler_output = self.sampler(logits=selected_logits,
                                                      sampling_metadata=sampling_metadata)
        selected_tokens = sampler_output.sampled_token_ids

        target_token_ids: torch.Tensor = selected_tokens[:num_decodes * (spec_k + 1)]
        draft_token_ids: torch.Tensor = draft_tokens[:num_decodes]
        target_token_ids_prefill_make = selected_tokens[num_decodes * (spec_k + 1):]

        # verify
        # (num_decodes + num_prefills, spec_k + 1)
        accepted_tokens = torch.full((num_decodes + num_prefills, spec_k + 1), -1, dtype=torch.int32, device=device)
        accepted_lens = torch.empty((num_decodes + num_prefills,), dtype=torch.long, device=device)
        if num_decodes:
            target_token_ids = target_token_ids.reshape(num_decodes, spec_k + 1)
            # only decode
            #target_probs = top_k_probs[:num_decodes * (spec_k + 1), :]
            #draft_probs = draft_probs[:num_decodes, :]
            draft_probs = None
            padded_spec_logits = selected_logits[:(num_decodes * (spec_k + 1)), :]
            accepted_tokens[:num_decodes, :], accepted_lens[:num_decodes] = \
                self.rejection_sampler.rejection_sampler_forward_with_fused_mtp(
                                    num_decodes,
                                    spec_k,
                                    draft_token_ids,
                                    draft_probs,
                                    padded_spec_logits,
                                    target_token_ids,
                                    sampling_metadata
                                )
                
                
        if num_prefills:
            accepted_tokens[num_decodes:, :] = torch.full((num_prefills, spec_k + 1), fill_value=-1, dtype=torch.long,
                                                          device=device)
            accepted_tokens[num_decodes:, 0] = target_token_ids_prefill_make.squeeze(-1)
            # (num_decodes + num_prefills, )
            accepted_lens[num_decodes:] = torch.full((num_prefills,), fill_value=1, dtype=torch.long, device=device)

        # Update input_ids
        if num_prefills == 0:
            # for requests in decoding stage, sample
            mtp_input_ids = accepted_tokens.flatten()  # [num_decodes x (spec_k + 1) , 1]
        else:
            # [num_all_input_tokens, 1]
            mtp_input_ids = torch.cat(
                (selected_tokens[:prefill_query_start], input_ids[prefill_query_start:].unsqueeze(-1)), dim=0)
            mtp_input_ids = mtp_input_ids.squeeze(-1)
        
        mtp_positions = positions
        mtp_hidden_states = ds_hidden_states
        if input_ids.shape[0] > prefill_query_start and num_prefills == 0:
            #somehow got padded
            mtp_positions = mtp_positions[:prefill_query_start]
            mtp_hidden_states = mtp_hidden_states[:prefill_query_start]
        
        draft_tokens = torch.full((num_decodes + num_prefills, spec_k), fill_value=0, dtype=torch.long, device=device)
        #draft_probs = torch.zeros((num_decodes + num_prefills, spec_k, 1), dtype=torch.float32, device=device)

        if num_decodes:
            # [num_decodes x 1]
            decode_accepted_lens = accepted_lens[:num_decodes]
            selected_token_index = (decode_accepted_lens - 1).type(torch.long).unsqueeze(-1)
            # we don't need draft probs for now
            #selected_probs_index = selected_token_index.unsqueeze(-1).expand(-1, -1, selected_logits.size(1))

        if num_decodes:
            for i in range(spec_k):
                # in prefill: [num_all_input_tokens, d_h]
                # in decode: [num_decodes x (spec_k + 1) , d_h]
                # in enflame solution, when ep is enabled, we don't care about dp_metadata.
                # so no need to keep input_tokens consistency with main model hence redundant computation is saved.
                if self.vllm_config.parallel_config.enable_expert_parallel:
                    if i == 0 and num_prefills > 0:
                        mtp_input_ids = mtp_input_ids[:prefill_query_start]
                        mtp_positions = mtp_positions[:prefill_query_start]
                        mtp_hidden_states = mtp_hidden_states[:prefill_query_start]
                    attn_metadata.num_prefills = 0
                    num_prefills = 0
                mtp_hidden_states = self.model.forward_mtp(
                    input_ids=mtp_input_ids,
                    positions=mtp_positions,
                    inputs_embeds=None,
                    previous_hidden_states=mtp_hidden_states,
                    spec_step_idx=i,
                )
                next_mtp_hidden_states = mtp_hidden_states
                # for sampling
                decode_hidden_states = mtp_hidden_states[:prefill_query_start]
                if num_prefills > 0:
                    prefill_hidden_states = torch.index_select(
                        mtp_hidden_states, dim=0, index=(query_start_loc[num_decodes + 1:] - 1))
                else:
                    prefill_hidden_states = torch.index_select(
                        ds_hidden_states, dim=0, index=(query_start_loc[num_decodes + 1:] - 1))
                mtp_hidden_states = torch.cat(
                    (decode_hidden_states, prefill_hidden_states), dim=0)

                # [num_decodes x (spec_k + 1) + num_prefill, vocab_size]
                # mtp_logits = self.simple_logits(mtp_hidden_states)
                mtp_logits = self.compute_logits(mtp_hidden_states, is_mtp_layer=True)
                # [num_decodes x (spec_k + 1) + num_prefill]
                #mtp_token_ids, mtp_probs = self.sample_ds(logits=mtp_logits,
                #                                             sampling_metadata=sampling_metadata)
                # [num_decodes, spec_k], 丢弃prefill
                #mtp_probs = mtp_probs[:num_decodes * (spec_k + 1), :].reshape(num_decodes, spec_k + 1,
                #                                                              mtp_logits.shape[-1])
                # sampler_output = self.sampler(logits=mtp_logits,
                #                                       sampling_metadata=sampling_metadata)
                # mtp_token_ids = sampler_output.sampled_token_ids
                mtp_token_ids = torch.argmax(mtp_logits, dim=-1, keepdim=True)
                accepted_mtp_tokens = mtp_token_ids[:num_decodes * (spec_k + 1)].reshape(num_decodes,
                                                                                         spec_k + 1).gather(dim=1,
                                                                                                            index=selected_token_index)
                # 兼容第一次decode, we don't need draft probs for now
                #mask = accepted_mtp_tokens.eq(-1).to(torch.bool)
                #mtp_tokens = torch.where(mask, 0, accepted_mtp_tokens)
                #draft_probs[:num_decodes, i, :] = mtp_probs.gather(1, selected_probs_index).gather(dim=-1,
                #                                                                                   index=mtp_tokens.unsqueeze(
                #                                                                                       -1).to(
                #                                                                                       torch.long)).squeeze(
                #    1)
                draft_tokens[:num_decodes, i] = accepted_mtp_tokens.squeeze(1)
                
                if spec_k == 1:
                    break
                # Todo(guozelin) fix spec_k > 1
                # don't bother with spec_k > 1 for now
                # update input_ids and positions for decode, keep unchangeable for prefill
                #mtp_input_ids[:num_decodes * (spec_k + 1)] = mtp_token_ids[:num_decodes * (spec_k + 1)]
                #mtp_positions[:num_decodes * (spec_k + 1)] += 1
                #if not self.use_dp:
                #    mtp_hidden_states = next_mtp_hidden_states
                #else:
                #    # repeat for dp
                #    ds_prefill_hidden_states = ds_hidden_states[query_start_loc[num_decodes]:]
                #    mtp_hidden_states = torch.concat((decode_hidden_states, ds_prefill_hidden_states),
                #                                                      dim=0)

                #if i < (spec_k - 1):
                #    if i == 0 and not self.use_dp:  # mtp > 1
                #        # split hidden_states for sampling
                #        attn_metadata.num_prefills = 0
                #        num_prefills = 0
                #        attn_metadata.num_actual_tokens = prefill_query_start
                #        mtp_input_ids = mtp_input_ids[:prefill_query_start]
                #        mtp_positions = mtp_positions[:prefill_query_start]
                #        mtp_hidden_states = next_mtp_hidden_states[:prefill_query_start]
                    # update decode metadata
                #    if not self.use_dp:
                #        attn_metadata.slot_mapping = (
                #        attn_metadata.slot_mapping[:prefill_query_start] + 1)
                #    else:
                #        attn_metadata.slot_mapping[:prefill_query_start] += 1

                #    attn_metadata.decode.input_positions += 1
                #    attn_metadata.decode.seq_lens += 1
                #    num_q_heads = self.num_query_heads
                #    mla_seq_lens = attn_metadata.decode.seq_lens
                #    # only fused mtp
                #    num_sq = spec_k + 1
                #    mla_seq_lens = torch.squeeze(torch.reshape(mla_seq_lens, (-1, num_sq))[:, num_sq - 1:], -1)
                #    num_q_heads = num_q_heads * num_sq
                #    get_mla_metadata(
                #        mla_seq_lens,
                #        num_q_heads,
                #        1,  # MQA for the decode path
                #        tile_scheduler_metadata=attn_metadata.decode.tile_scheduler_metadata,
                #        num_splits=attn_metadata.decode.num_splits[:mla_seq_lens.shape[0] + 1],
                #    )
        else:
            # prefill spec_k times for dp mock request.
            iter = spec_k if self.use_dp else 1
            for i in range(iter):
                self.model.forward_mtp(
                    input_ids = mtp_input_ids,
                    positions = mtp_positions,
                    inputs_embeds = None,
                    previous_hidden_states = mtp_hidden_states,
                    spec_step_idx = 0
                )

        return selected_tokens, accepted_tokens, accepted_lens, draft_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        draft_tokens: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        top_p: Optional[torch.Tensor] = None,
        top_k: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        if not self.vllm_config.additional_config["deepseek_fused_mtp"]:
            return hidden_states
        if self.use_torch_compile:
            main_model_sampled_tokens, accepted_tokens, accepted_lens, draft_tokens, next_token_ids = \
                torch.ops.vllm.fused_mtp(hidden_states, input_ids, positions, temperature,
                           top_p, top_k, draft_tokens, self.layer_name)
        else:    
            main_model_sampled_tokens, accepted_tokens, accepted_lens, draft_tokens, next_token_ids = \
                self.fused_mtp(hidden_states, input_ids, positions, temperature,
                           top_p, top_k, draft_tokens)
        
        return IntermediateTensors({
            # [(num_decodes x (spec_k + 1)) + num_prefills, 1]
            "main_model_sampled_tokens" : main_model_sampled_tokens,
            # (num_decodes + num_prefills, spec_k + 1)
            "accepted_tokens": accepted_tokens,
            # (num_decodes + num_prefills, )
            "accepted_lens": accepted_lens,
            # (num_decodes + num_prefills, spec_k)
            "next_draft_tokens": draft_tokens,
            # (num_decodes + num_prefills, spec_k + 1)
            "next_token_ids": next_token_ids,
        })

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        is_mtp_layer: bool = False
    ) -> Optional[torch.Tensor]:

        if is_mtp_layer and self.vllm_config.additional_config["deepseek_fused_mtp"]:
            logits = self.logits_processor(self.model.layers[-1].shared_head.head,
                                           self.model.layers[-1].shared_head(hidden_states))
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # TODO: should we check all layers?
        if len(self.model.layers) > self.model.start_layer and \
                getattr(self.model.layers[self.model.start_layer].self_attn, 'qkv_fuse', False):
            stacked_params_mapping += [
                ("qkv_a_proj_with_mqa", "q_a_proj", 0),
                ("qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", 1),
            ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping_main = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

        expert_params_mapping_mtp = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "mlp.shared_experts" in name and name not in params_dict:
                if self.quant_config is not None:
                    name = name.replace(
                        "mlp.shared_experts", "mlp.shared_experts._orig_mod"
                    )

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                if not gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
                    continue # skip spec decode layers for main model
                name = _rewrite_spec_layer_name(spec_layer, name)

            expert_params_mapping = expert_params_mapping_mtp if spec_layer is not None else expert_params_mapping_main

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    if name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


# Compatibility with
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/configuration_deepseek.py
def get_spec_layer_idx_from_weight_name(config,
                                        weight_name: str) -> Optional[int]:
    if (hasattr(config, "num_nextn_predict_layers")
            and config.num_nextn_predict_layers > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None

def _rewrite_spec_layer_name(spec_layer: int, name: str) -> str:
    """
    Rewrite the weight name to match the format of the original model.
    Add .mtp_block for modules in transformer layer block for spec layer
    and rename shared layer weights to be top level.
    """
    spec_layer_weight_names = [
        "embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"
    ]
    spec_layer_weight = False
    for weight_name in spec_layer_weight_names:
        if weight_name in name:
            spec_layer_weight = True
            break
    if not spec_layer_weight:
        # treat rest weights as weights for transformer layer block
        name = name.replace(f"model.layers.{spec_layer}.",
                            f"model.layers.{spec_layer}.mtp_block.")
    return name

def fused_mtp(ds_hidden_states: torch.Tensor,
              input_ids: torch.Tensor,
              positions: torch.Tensor,
              temperature: torch.Tensor,
              top_p: torch.Tensor,
              top_k: torch.Tensor,
              draft_tokens: torch.Tensor,
              layer_name: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.fused_mtp(ds_hidden_states, input_ids, positions, temperature, top_p, top_k, draft_tokens)

def fused_mtp_fake(ds_hidden_states: torch.Tensor,
              input_ids: torch.Tensor,
              positions: torch.Tensor,
              temperature: torch.Tensor,
              top_p: torch.Tensor,
              top_k: torch.Tensor,
              draft_tokens: torch.Tensor,
              layer_name: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(input_ids), \
                   torch.empty((draft_tokens.shape[0], draft_tokens.shape[-1] + 1), dtype = torch.int32, device = draft_tokens.device),\
                   torch.empty((draft_tokens.shape[0]), dtype = torch.int32, device = draft_tokens.device), \
                   torch.empty_like(draft_tokens), \
                   torch.empty((draft_tokens.shape[0], 1), dtype = torch.int32, device = draft_tokens.device)


direct_register_custom_op(
    op_name="fused_mtp",
    op_func=fused_mtp,
    mutates_args=[],
    fake_impl=fused_mtp_fake,
    dispatch_key=current_platform.dispatch_key,
)