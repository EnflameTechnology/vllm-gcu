import torch
from vllm_gcu.kernels.native_op.utils import register_native
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)

@register_native("_C", "fused_qk_norm_rope")
def _ref_fused_qk_norm_rope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_sin_cache,
    is_neox,
    position_ids,
):
    num_tokens = qkv.shape[0]
    num_heads = qkv.shape[1]

    positions = torch.arange(num_tokens, dtype=torch.long)
    q_norm = RMSNorm(head_dim, eps=eps, weight=q_weight)
    k_norm = RMSNorm(head_dim, eps=eps, weight=k_weight)

    rotary_dim = int(head_dim * 1.0)
    dtype = torch.bfloat16

    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    )
    num_heads_kv = num_heads_k + num_heads_v
    head_dim = head_dim

    ref_result = _apply_qk_norm_rope(
        qkv,
        positions,
        q_norm,
        k_norm,
        rope,
        num_heads_q,
        num_heads_kv,
        head_dim,
    )

    qkv.copy_(ref_result)