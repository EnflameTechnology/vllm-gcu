# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# coding=utf-8
# Copyright 2024 The HunYuan team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only HunYuan model compatible with HuggingFace weights."""

# 该补丁将会在下个vllm_gcu 版本移除

import types
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import regex as re
import torch
from torch import nn
from vllm.config import  VllmConfig
from vllm.distributed import (
    get_pp_group,

)
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)

from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    maybe_prefix,
    is_pp_missing_parameter
)
from vllm.model_executor.models.hunyuan_v1 import(
    HunYuanModel,
    HunYuanDecoderLayer,
    HunYuanSparseMoeBlock,
    _get_cla_factor
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)

class HunyuanV1ModelBase(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = HunYuanModel(vllm_config=vllm_config, prefix="model")
        self.model.load_weights = types.MethodType(load_weights, self.model)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, config.vocab_size, logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)


class HunYuanMoEV1Base(HunyuanV1ModelBase, MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Set MoE hyperparameters
        self.expert_weights = []
        self.num_expert_groups = 1
        self.moe_layers: list[SharedFusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, HunYuanDecoderLayer)
            if isinstance(layer.mlp, HunYuanSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No HunYuanMoE layer found in model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            self.expert_weights.append(layer.get_expert_weights())
            # Register the expert weights.
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, HunYuanSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class HunYuanDenseV1Base(HunyuanV1ModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class HunYuanDenseV1ForCausalLM(HunYuanDenseV1Base):
    pass


class HunYuanMoEV1ForCausalLM(HunYuanMoEV1Base):
    pass


def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
    cla_factor = _get_cla_factor(self.config)
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]

    num_attention_heads = self.config.num_attention_heads
    num_kv_heads = getattr(self.config, "num_key_value_heads",
                            self.config.num_attention_heads)
    split_params_mapping = [
        (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
        (
            ".qkv_proj",
            ".qkv_proj",
            num_attention_heads + num_kv_heads * 2,
            [("q", num_attention_heads), ("k", num_kv_heads),
                ("v", num_kv_heads)],
            self._split_qkv_weight,
        ),
    ]

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    expert_params_mapping = self.get_expert_mapping()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        if "gate_proj_bias" in name:
            name = name.replace("gate_proj_bias", "gate_proj.bias")
        if "up_proj_bias" in name:
            name = name.replace("up_proj_bias", "up_proj.bias")
        if ("rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name):
            # Models trained using ColossalAI may include these tensors in
            # the checkpoint. Skip them.
            continue
        # With tie_word_embeddings, we can skip lm_head.weight
        # The weight might appear unnecessarily in the files if the model is
        # processed with quantization, LoRA, fine-tuning, etc.
        if self.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)):
            # Loading kv cache scales for compressed-tensors quantization
            param = params_dict[scale_name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                loaded_weight[0])
            weight_loader(param, loaded_weight)
            loaded_params.add(scale_name)
            continue

        is_found = False
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if "mlp.experts" in name:
                continue
            # cross layer only have q_proj, skip qkv pack
            if weight_name == ".q_proj":
                match = re.search(r"layers\.\d+", name)
                if match:
                    layer_id = int(match.group(0).split(".")[-1])
                    if cla_factor > 1 and layer_id % cla_factor != 0:
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
            loaded_params.add(name)
            is_found = True
            break
        if is_found:
            continue

        for (
                param_name,
                weight_name,
                den,
                split_param,
                func,
        ) in split_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            assert loaded_weight.shape[0] % den == 0
            units = loaded_weight.shape[0] // den

            param = params_dict[name]
            weight_loader = param.weight_loader
            offset = 0
            for shard_id, num in split_param:
                new_offset = offset + num * units
                if func:
                    weight_loader(param,
                                    func(loaded_weight)[offset:new_offset],
                                    shard_id)
                else:
                    weight_loader(param, loaded_weight[offset:new_offset],
                                    shard_id)
                offset = new_offset

            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                # this is an expert weight and should not be
                # attempted to load as other weights later
                is_expert_weight = True

                # Do not modify `name` since the loop may continue here
                # Instead, create a new variable
                name_mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                param = params_dict[name_mapped]
                # We should ask the weight loader to return success or not
                # here since otherwise we may skip experts with other
                # available replicas.
                weight_loader = typing.cast(Callable[..., bool],
                                            param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    name = name_mapped
                    break
            else:
                if is_expert_weight:
                    # We've checked that this is an expert weight
                    # However it's not mapped locally to this rank
                    # So we simply skip it
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if "mlp.gate.wg." in name:
                    name = name.replace("wg.", "")

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params