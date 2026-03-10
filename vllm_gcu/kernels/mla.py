#!/usr/bin/env python
# coding=utf-8
import torch

from vllm.attention.layer import Attention
from vllm.config import CacheConfig
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.distributed import get_tp_group
from vllm_gcu.utils import scatter
from vllm.model_executor.custom_op import CustomOp
import vllm_gcu.envs as gcu_envs


@MultiHeadLatentAttention.register_oot
class GCUMultiHeadLatentAttention(MultiHeadLatentAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        CustomOp.__init__(self)
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        fusion_args = {}
        if self.indexer and self.is_sparse and gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
            fusion_args = {
                'rotary_emb':mla_modules.rotary_emb,
                'kv_a_layernorm':mla_modules.kv_a_layernorm,
            }
        self.mla_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=scale,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            use_sparse=self.is_sparse,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_b_proj,
            indexer=self.indexer,
            # Fusion
            **fusion_args,
        )

        self.prefix = prefix

    def forward_oot(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        # llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        indexer_k = None

        tp_group = get_tp_group()
        world_size = tp_group.world_size
        need_gather = world_size > 1 and hidden_states.shape[0] != positions.shape[0]
        if need_gather:
            assert self.indexer is None
            seqlen = len(positions)
            scatter_counts = scatter(seqlen, world_size)

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            if need_gather:
                qkv_lora = tp_group.all_gatherv(qkv_lora, dim=0, sizes=scatter_counts)
            if self.indexer and self.is_sparse and gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
                indexer_k, q_c, kv_lora = qkv_lora.split(
                    [self.indexer.head_dim, self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            else:
                q_c, kv_lora = qkv_lora.split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            if need_gather:
                hidden_states = tp_group.all_gatherv(
                    hidden_states, dim=0, sizes=scatter_counts
                )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        q = q.view(-1, self.num_heads, self.qk_head_dim)

        if self.indexer and self.is_sparse:  # v3.2
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb, indexer_k,
            )

        # if llama_4_scaling is not None:
        #     q *= llama_4_scaling
        if self.indexer and self.is_sparse and gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
            attn_out = self.mla_attn(
                q,
                kv_lora,
                positions,
                output_shape=(q.shape[0], self.num_heads * self.v_head_dim),
            )
        else:
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c)
            k_pe = k_pe.unsqueeze(1)
            if self.rotary_emb is not None:
                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe
                )
            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            )

        return self.o_proj(attn_out)[0]