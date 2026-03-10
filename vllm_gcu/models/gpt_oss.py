import torch

from typing import Optional

from unittest.mock import patch

from vllm.model_executor.models.gpt_oss import GptOssForCausalLM
from vllm.model_executor.layers.quantization import QuantizationConfig
from transformers import GptOssConfig
from vllm.config import CacheConfig

from vllm.model_executor.models.gpt_oss import OAIAttention
from vllm.model_executor.layers.rotary_embedding import get_rope


class OAIAttentionCustomed(OAIAttention):

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=prefix,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            # dtype=torch.float32,
            rope_scaling={
                "rope_type": "yarn",
                "factor": config.rope_scaling["factor"],
                "original_max_position_embeddings": config.rope_scaling[
                    "original_max_position_embeddings"
                ],
                "beta_fast": config.rope_scaling["beta_fast"],
                "beta_slow": config.rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )


patch(
    "vllm.model_executor.models.gpt_oss.OAIAttention", OAIAttentionCustomed
).start()
