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

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP, SupportsLoRA
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
from vllm_gcu.kernels.sampler_fusedmtp import FusedMTPSampler as SamplerDS
from vllm_gcu.kernels.sampler import ParallelTopKTopPSampler
from vllm_gcu.kernels.rejection_sampler import GCURejectionSampler as RejectionSamplerDS
#from vllm.v1.sample.rejection_sampler import RejectionSampler as RejectionSamplerDS
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op
from vllm_gcu.attention.backends.mla_v1 import GCUMLADecodeMetadata
from vllm_gcu.models.deepseek_v3.deepseek_v3 import DeepseekV2MLP, DeepseekV2MoE
from vllm_gcu.models.deepseek_v32 import DeepseekV2Attention, DeepseekV2MLAAttention, DeepseekV2DecoderLayer, Indexer


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


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0



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
        topk_indices_buffer: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
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
        self.mtp_block = DeepseekV2DecoderLayer(vllm_config, prefix, topk_indices_buffer, False)
        
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
        self.config = config

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device="cuda")
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(vllm_config, prefix,
                                                  topk_indices_buffer,
                                                  vllm_config.parallel_config.enable_eplb),
            prefix=f"{prefix}.layers",
        )
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 0)
        if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
            mtp_layers = torch.nn.ModuleDict({
                str(idx):
                    DeepSeekMultiTokenPredictorLayer(
                        vllm_config,
                        f"{prefix}.layers.{idx}",
                        topk_indices_buffer
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
class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts,
                            SupportsLoRA):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # `packed_modules_mapping` needs to be modified before
        # initializing DeepseekV2Model, as it is passed inplace to
        # quantization config init and may be used to select the
        # quant_method for relevant layers during initialization.
        self.fuse_qkv_a_proj = hasattr(
            config, "q_lora_rank") and config.q_lora_rank is not None
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

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

        self.model = DeepseekV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
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

        self.sampler = SamplerDS(spec_k=self.num_speculative_tokens)
        logprobs_mode = self.sampler.topk_topp_sampler.logprobs_mode
        self.sampler.topk_topp_sampler = ParallelTopKTopPSampler(logprobs_mode)
        self.rejection_sampler = RejectionSamplerDS()
        self.use_dp = vllm_config.parallel_config.data_parallel_size > 1
        
        max_num_seq = vllm_config.scheduler_config.max_num_seqs
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        device = torch.gcu.current_device()
        
        self.accepted_tokens = torch.full((max_num_seq, self.num_speculative_tokens + 1), fill_value=-1, dtype=torch.int32, device=device)
        self.accepted_lens = torch.full((max_num_seq, ),  1, dtype=torch.int32, device=device)
        self.draft_tokens = torch.full((max_num_seq, self.num_speculative_tokens),  0, dtype=torch.int32, device=device)
        self.mtp_input_ids = torch.empty(max_num_batched_tokens, dtype=torch.int32, device=device)
        self.arange = torch.arange(max_num_seq+1, device=device, dtype=torch.int32)
    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
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
        self.num_redundant_experts = (num_physical_experts -
                                      self.num_logical_experts)
        for layer in self.model.layers:
            if isinstance(layer.mlp, DeepseekV2MoE):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)
    
    def fused_mtp(self,
                  ds_hidden_states,
                  input_ids,
                  positions,
                  temperature,
                  top_p,
                  top_k,
                  repetition_penalty,
                  frequency_penalty,
                  presence_penalty,
                  prompt_token_ids,
                  output_token_ids,
                  draft_tokens,
                  logits_indices):
        
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
                temperature, top_p, top_k, repetition_penalty,
                frequency_penalty, presence_penalty, prompt_token_ids,
                output_token_ids, draft_tokens, attn_metadata)
            main_model_sampled_tokens = main_model_sampled_tokens.squeeze(-1)
        else:
            _, accepted_tokens, accepted_lens, draft_tokens = self.fused_mtp_prefill_decode(
                ds_hidden_states, input_ids, positions,
                temperature, top_p, top_k, repetition_penalty,
                frequency_penalty, presence_penalty, prompt_token_ids,
                output_token_ids, draft_tokens, attn_metadata, logits_indices=logits_indices)
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
                                 repetition_penalty,
                                 frequency_penalty,
                                 presence_penalty,
                                 prompt_token_ids,
                                 output_token_ids,
                                 draft_tokens,
                                 attn_metadata):
        spec_k = self.num_speculative_tokens
        bsz = ds_hidden_states.shape[0] // (spec_k + 1)
        selected_hidden_states = ds_hidden_states
        query_start_loc = attn_metadata.query_start_loc
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
            no_penalties=not self.vllm_config.additional_config.get("deepseek_fused_mtp_use_penalty", True),
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=frequency_penalty,
            presence_penalties=presence_penalty,
            repetition_penalties=repetition_penalty,
            output_token_ids=output_token_ids,
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
        decode_last_sample = query_start_loc[:-1] + accepted_lens - 1

        mtp_hidden_states = self.model.forward_mtp(
            input_ids=mtp_input_ids.to(torch.int32),
            positions=mtp_positions,
            inputs_embeds=None,
            previous_hidden_states=mtp_hidden_states,
            spec_step_idx=0,
        )
        mtp_logits = self.compute_logits(mtp_hidden_states, is_mtp_layer=True)

        mtp_token_ids = torch.argmax(mtp_logits, dim=-1, keepdim=True)
        draft_token = torch.index_select(mtp_token_ids, dim=0, index=decode_last_sample).squeeze(-1)
        draft_tokens[:, 0] = draft_token
            
        # loops for spec_k > 1; we only support spec_k = 1 & 2 for now
        for i in range(1, spec_k):
            mtp_input_ids = draft_token
            mtp_positions += 1
            attn_metadata.slot_mapping += 1
            if i == 1:
                mtp_positions = torch.index_select(mtp_positions, dim=0, index=decode_last_sample)
                mtp_hidden_states = torch.index_select(mtp_hidden_states, dim=0, index=decode_last_sample)
                attn_metadata.slot_mapping = torch.index_select(attn_metadata.slot_mapping, dim=0, index=decode_last_sample)
            attn_metadata.decode.seq_lens += 1
            attn_metadata.decode.max_decode_seq_len += 1
            attn_metadata.max_query_len = 1
            attn_metadata.query_start_loc = self.arange[:bsz+1]
            attn_metadata.num_actual_tokens = bsz
            attn_metadata.num_decode_tokens = bsz

            # mtpstep2
            mtp_hidden_states = self.model.forward_mtp(
                input_ids=mtp_input_ids.to(torch.int32),
                positions=mtp_positions,
                inputs_embeds=None,
                previous_hidden_states=mtp_hidden_states,
                spec_step_idx=i,
            )
            mtp_logits = self.compute_logits(mtp_hidden_states, is_mtp_layer=True)

            mtp_token_ids = torch.argmax(mtp_logits, dim=-1, keepdim=True)
            draft_token = mtp_token_ids.squeeze(-1)
            draft_tokens[:, i] = draft_token
        return selected_tokens, accepted_tokens, accepted_lens, draft_tokens

    def fused_mtp_prefill_decode(self,
                                 ds_hidden_states,
                                 input_ids,
                                 positions,
                                 temperature,
                                 top_p,
                                 top_k,
                                 repetition_penalty,
                                 frequency_penalty,
                                 presence_penalty,
                                 prompt_token_ids,
                                 output_token_ids,
                                 draft_tokens,
                                 attn_metadata,
                                 logits_indices):
        device = ds_hidden_states.device
        spec_k = self.num_speculative_tokens
        dtype = ds_hidden_states.dtype
        # adjust for mla（和 attention builder 中呼应）
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decodes
        # [num_decodes + num_prefills + 1]
        query_start_loc = attn_metadata.query_start_loc
        prefill_query_start = num_decodes * (spec_k + 1)

        selected_hidden_states = torch.index_select(
            ds_hidden_states, dim=0, index=logits_indices)

        # [(num_decodes x (spec_k + 1)) + num_prefills , vocab_size]
        selected_logits = self.compute_logits(selected_hidden_states)
        sampling_metadata = SamplingMetadataDs(
            temperature=temperature,
            all_greedy=False,
            all_random=False,
            top_p=top_p,
            top_k=top_k,
            generators={},
            max_num_logprobs=None,
            no_penalties=not self.vllm_config.additional_config.get("deepseek_fused_mtp_use_penalty", True),
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=frequency_penalty,
            presence_penalties=presence_penalty,
            repetition_penalties=repetition_penalty,
            output_token_ids=output_token_ids,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        sampler_output = self.sampler(logits=selected_logits,
                                                      sampling_metadata=sampling_metadata)
        selected_tokens = sampler_output.sampled_token_ids
        
        num_decode_tokens = num_decodes * (spec_k + 1)
        
        target_token_ids = selected_tokens[:num_decode_tokens]
        draft_token_ids = draft_tokens[:num_decodes]

        # verify
        # (num_decodes + num_prefills, spec_k + 1)
        accepted_tokens = self.accepted_tokens[:num_decodes + num_prefills].fill_(-1)
        accepted_lens = self.accepted_lens[:num_decodes + num_prefills].fill_(1)

        if num_decodes:
            target_token_ids = target_token_ids.reshape(num_decodes, spec_k + 1)
            draft_probs = None
            padded_spec_logits = selected_logits[:num_decode_tokens, :]
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

        accepted_tokens[num_decodes:, 0] = selected_tokens[num_decode_tokens:, 0]
        prefill_last_sample = query_start_loc[num_decodes + 1:] - 1
        
        mtp_input_ids = self.mtp_input_ids[:len(input_ids)]
        
        mtp_input_ids[num_decode_tokens:-1] = input_ids[num_decode_tokens + 1:]

        # mtp_input_ids[prefill_last_sample] = accepted_tokens[num_decodes:, 0]

        mtp_input_ids.scatter_(dim=0, index=prefill_last_sample, src=accepted_tokens[num_decodes:, 0] )
        
        mtp_input_ids[:num_decode_tokens] = accepted_tokens[:num_decodes].flatten()

        mtp_positions = positions
        
        mtp_hidden_states = ds_hidden_states
        
        draft_tokens = self.draft_tokens[:num_decodes + num_prefills].fill_(0)

        # mtp step 1
        mtp_hidden_states = self.model.forward_mtp(
            input_ids=mtp_input_ids.to(torch.int32),
            positions=mtp_positions,
            inputs_embeds=None,
            previous_hidden_states=mtp_hidden_states,
            spec_step_idx=0,
        )
        decode_last_sample = query_start_loc[:num_decodes] + accepted_lens[:num_decodes] - 1
        mtp_logits_index = torch.cat((decode_last_sample, prefill_last_sample))
        mtp_hidden_states = torch.index_select(mtp_hidden_states, dim=0, index=mtp_logits_index)
        mtp_logits = self.compute_logits(mtp_hidden_states, is_mtp_layer=True)
        mtp_token_ids = torch.argmax(mtp_logits, dim=-1, keepdim=True)
        draft_token = mtp_token_ids.squeeze(-1)
        draft_tokens[:,0] = draft_token

        # loops for spec_k > 1; we only support spec_k = 1 & 2 for now
        for i in range(1, spec_k):
            mtp_input_ids = draft_token
            mtp_positions += 1
            attn_metadata.slot_mapping += 1
            if i == 1:
                mtp_positions = torch.index_select(mtp_positions, dim=0, index=mtp_logits_index)
                attn_metadata.slot_mapping = torch.index_select(attn_metadata.slot_mapping, dim=0, index=mtp_logits_index)
            attn_metadata.max_query_len = 1
            attn_metadata.query_start_loc = self.arange[:num_decodes+num_prefills+1]
            attn_metadata.num_actual_tokens = num_decodes+num_prefills
            attn_metadata.num_decode_tokens = num_decodes+num_prefills
            if num_decodes:
                attn_metadata.decode.seq_lens += 1
                attn_metadata.decode.max_decode_seq_len += 1
                if num_prefills:
                    step_k_query_start_loc = attn_metadata.prefill.query_start_loc
                    step_k_seq_lens = step_k_query_start_loc[1:] - step_k_query_start_loc[:-1] + 1
                    attn_metadata.decode.seq_lens = torch.cat((attn_metadata.decode.seq_lens, step_k_seq_lens))
                    attn_metadata.decode.block_table = torch.cat((attn_metadata.decode.block_table, attn_metadata.prefill.block_table),0)
            else:
                step_k_query_start_loc = attn_metadata.prefill.query_start_loc
                step_k_block_table = attn_metadata.prefill.block_table
                step_k_seq_lens = step_k_query_start_loc[1:] - step_k_query_start_loc[:-1] + 1
                attn_metadata.decode = GCUMLADecodeMetadata(
                    block_table=step_k_block_table,
                    seq_lens=step_k_seq_lens,
                    max_decode_seq_len=step_k_seq_lens.max().item(),
                    tile_scheduler_metadata=None,
                    num_splits=None
                )
            num_decodes += num_prefills
            num_prefills = 0
            attn_metadata.num_decodes = num_decodes
            attn_metadata.num_prefills = num_prefills
            attn_metadata.prefill = None
            mtp_hidden_states = self.model.forward_mtp(
                input_ids=mtp_input_ids.to(torch.int32),
                positions=mtp_positions,
                inputs_embeds=None,
                previous_hidden_states=mtp_hidden_states,
                spec_step_idx=i,
            )
            mtp_hidden_states_for_logits = mtp_hidden_states
            mtp_logits = self.compute_logits(mtp_hidden_states_for_logits, is_mtp_layer=True)
            mtp_token_ids = torch.argmax(mtp_logits, dim=-1, keepdim=True)
            draft_token = mtp_token_ids.squeeze(-1)
            draft_tokens[:,i] = draft_token
        
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
        repetition_penalty: Optional[torch.Tensor] = None,
        frequency_penalty: Optional[torch.Tensor] = None,
        presence_penalty: Optional[torch.Tensor] = None,
        prompt_token_ids: Optional[torch.Tensor] = None,
        output_token_ids: Optional[torch.Tensor] = None,
        logits_indices: Optional[torch.Tensor] = None,
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
                           top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty,
                           prompt_token_ids, output_token_ids, draft_tokens, logits_indices, self.layer_name)
        else:    
            main_model_sampled_tokens, accepted_tokens, accepted_lens, draft_tokens, next_token_ids = \
                self.fused_mtp(hidden_states, input_ids, positions, temperature,
                           top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty,
                           prompt_token_ids, output_token_ids, draft_tokens, logits_indices)
        
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
            "hidden_states": hidden_states,
            "logist": torch.empty_like(hidden_states)
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
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
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if ((param_name == "fused_qkv_a_proj")
                        and name_mapped not in params_dict):
                    continue
                else:
                    name = name_mapped

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
              repetition_penalty: torch.Tensor,
              frequency_penalty: torch.Tensor,
              presence_penalty: torch.Tensor,
              prompt_token_ids: torch.Tensor,
              output_token_ids: torch.Tensor,
              draft_tokens: torch.Tensor,
              logits_indices: torch.Tensor,
              layer_name: str = "",
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.fused_mtp(ds_hidden_states, input_ids, positions, temperature, top_p, top_k, repetition_penalty,
                          frequency_penalty, presence_penalty, prompt_token_ids, output_token_ids, draft_tokens, logits_indices)

def fused_mtp_fake(ds_hidden_states: torch.Tensor,
              input_ids: torch.Tensor,
              positions: torch.Tensor,
              temperature: torch.Tensor,
              top_p: torch.Tensor,
              top_k: torch.Tensor,
              repetition_penalty: torch.Tensor,
              frequency_penalty: torch.Tensor,
              presence_penalty: torch.Tensor,
              prompt_token_ids: torch.Tensor,
              output_token_ids: torch.Tensor,
              draft_tokens: torch.Tensor,
              logits_indices: torch.Tensor,
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
