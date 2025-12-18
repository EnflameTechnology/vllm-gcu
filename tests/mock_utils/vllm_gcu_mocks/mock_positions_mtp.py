# SPDX-License-Identifier: Apache-2.0
from typing import Iterable, Optional
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import maybe_prefix
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm_gcu.models.deepseek_mtp import DeepSeekMultiTokenPredictorLayer


@support_torch_compile
class MockMTPModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        # NOTE: we need an attention layer for meta
        self.layers = torch.nn.ModuleDict({
            str(idx):
            DeepSeekMultiTokenPredictorLayer(
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds[:, 0] = positions.to(inputs_embeds.dtype)
        return inputs_embeds

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        positions = hidden_states[:, 0].to(torch.int64)
        # NOTE: mtp use pos n to infer n+2, we use pos + 1 to simulate reject
        # and pos + 2 for accept
        torch.randint_like(positions, low=1, high=3)
        logits = torch.nn.functional.one_hot(
            positions + torch.randint_like(positions, low=1, high=3),
            self.vocab_size)
        return logits.to(hidden_states.dtype)


class MockPosMTP(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = MockMTPModel(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))
        self.vllm_config = vllm_config

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, hidden_states,
                                   inputs_embeds, spec_step_idx)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[""],
        )
        return loader.load_weights(weights)
