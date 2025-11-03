# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only Deepseek model."""
from collections.abc import Iterable
import typing
from typing import Optional
from collections.abc import Iterable, Callable

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
import vllm.envs as envs
from vllm.model_executor.models.deepseek import (
    DeepseekMLP,
    DeepseekModel,
    DeepseekForCausalLM
)
from vllm.model_executor.models.utils import is_pp_missing_parameter
from unittest.mock import patch
from vllm_gcu.models.deepseek_v3.deepseek_v3 import custom_backend


class DeepseekMoEGCU(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_log2phy=None,
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}.")
        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.is_sequence_parallel = (envs.VLLM_ALL2ALL_BACKEND
                                     in ("deepep_high_throughput",
                                         "deepep_low_latency")
                                     and parallel_config.enable_expert_parallel
                                     and self.tp_size > 1)
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

            self.shared_experts = DeepseekMLP(
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
                routed_scaling_factor=1.0,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )

        self.tp_size = self.experts.tp_size

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


class DeepseekModelGCU(DeepseekModel):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
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
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DeepseekForCausalLMGCU(DeepseekForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        patch("vllm.model_executor.models.deepseek.DeepseekMoE", DeepseekMoEGCU).start()
        patch("vllm.model_executor.models.deepseek.DeepseekModel", DeepseekModelGCU).start()
        super().__init__(vllm_config=vllm_config, prefix=prefix)


