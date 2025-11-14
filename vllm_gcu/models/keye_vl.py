import torch
from vllm_gcu.kernels import _custom_ops as ops
from vllm.model_executor.models.keye_vl1_5 import KeyeVL1_5ForConditionalGeneration

from unittest.mock import patch

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_shape = q.shape
    k_shape = k.shape
    head_size = q_shape[-1]
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    rotary_pos_emb = torch.concat([cos,sin],-1).to(q.dtype)
    q = q.view([rotary_pos_emb.shape[0], -1])
    k = k.view([rotary_pos_emb.shape[0], -1])
    rotary_dim = cos.shape[0]
    positions = torch.arange(rotary_dim, device=q.device,dtype=torch.long)
    ops.rotary_embedding(
        positions,
        q,
        k,
        head_size,
        rotary_pos_emb,
        True
    )
    q = q.view(q_shape)
    k = k.view(k_shape)

    return q, k


patch("vllm.model_executor.models.keye.apply_rotary_pos_emb_flashatt", apply_rotary_pos_emb).start()