import torch

from unittest.mock import patch
from vllm.model_executor.layers.rotary_embedding import Phi3LongRoPEScaledRotaryEmbedding
from typing import Optional
from vllm.model_executor.layers.rotary_embedding.common import rotate_neox
from vllm_gcu.kernels import _custom_ops as ops


class Phi3LongRoPEScaledRotaryEmbeddingPatched(Phi3LongRoPEScaledRotaryEmbedding):
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert key is not None

        k = self.original_max_position_embeddings
        long_prompt_offset = (torch.any(positions > k).float() *
                              torch.full_like(positions, k)).long()
        idx = (torch.add(positions, long_prompt_offset)
               if long_prompt_offset is not None else positions)
        idx = torch.add(idx, offsets) if offsets is not None else idx
        ops.rotary_embedding(
                idx,
                query,
                key,
                self.head_size,
                self.long_short_cos_sin_cache,
                True
            )

        return query, key

patch("vllm.model_executor.layers.rotary_embedding.Phi3LongRoPEScaledRotaryEmbedding", 
      Phi3LongRoPEScaledRotaryEmbeddingPatched).start()