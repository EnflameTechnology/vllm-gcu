from unittest.mock import patch

import torch

def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = self.cos_sin_cache[:seqlen]
    return cos_sin

patch("vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.get_cos_sin", get_cos_sin, create=True).start()