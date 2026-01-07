# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionLayer
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm_gcu.attention.backends.flashmla_sparse import FlashMLASparseImpl, FlashMLASparseBackend, FlashMLASparseMetadata
from vllm_gcu.attention.backends.mla_v1_fusion import RopeWithKVCache

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


class FlashMLASparseFusionBackend(FlashMLASparseBackend):

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseFusionImpl"]:
        return FlashMLASparseFusionImpl


class FlashMLASparseFusionImpl(FlashMLASparseImpl):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            topk_indice_buffer: Optional[torch.Tensor] = None,
            indexer: Optional["Indexer"] = None,
            rotary_emb: Optional[RotaryEmbedding] = None,
            kv_a_layernorm: Optional[torch.nn.Module] = None,
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, topk_indice_buffer,
                         indexer, **mla_args)
        self.rotary_emb = rotary_emb
        self.kv_a_layernorm = kv_a_layernorm
        self.rope_with_kvcache = RopeWithKVCache(self.rotary_emb,
                                                 self.kv_a_layernorm,
                                                 self.kv_lora_rank,
                                                 self.qk_rope_head_dim,
                                                 kv_cache_dtype)

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        kv_lora: torch.Tensor,  # key in unified attn
        positions: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        kv_c, k_pe = kv_lora.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_c_normed = self.kv_a_layernorm(kv_c)

        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for MLACommonImpl")

        if attn_metadata is None:
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs

        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)

        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # TODO: handle index / kv_cache correctly
        topk_indices_global = None

        q = torch.cat([ql_nope, q_pe], dim=-1)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if self.kv_cache_dtype != "fp8_ds_mla":
            attn_out = self._forward_bf16_kv(q, kv_cache, topk_indices_global,
                                             attn_metadata)
        else:
            attn_out = self._forward_fp8_kv(q, kv_cache, topk_indices_global,
                                            attn_metadata)

        self._v_up_proj(attn_out, out=output[:num_actual_toks])
        return output
