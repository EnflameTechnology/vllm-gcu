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

    def _k_up_proj(self, out, q_nope):
        B, N, P = q_nope.shape
        # Multiply (B, N, P) x (N, P, L) -> (B, N, L)
        torch.bmm(q_nope.transpose(0, 1), self.W_UK_T, out=out.transpose(0, 1))

    def _v_up_proj(self, x, out=None):
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
        B = x.shape[0]
        # Multiply (B, N, L) x (N, L, V) -> (B, N, V)
        out_shape = (B, self.num_heads, self.W_UV.shape[-1])
        if out is None:
            out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        else:
            out = out.reshape(out_shape)

        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        # maybe linear_copy when B is not contiguous
        torch.bmm(x.transpose(0, 1), self.W_UV, out=out.transpose(0, 1))
        # Convert from (B, N, V) to (B, N * V)
        return out.view(-1, self.num_heads * self.v_head_dim)

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
        kv_lora = kv_lora[:num_actual_toks, ...]
        positions = positions[:num_actual_toks, ...]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        q_concat = torch.empty(
            (q.shape[0], self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        self._k_up_proj(q_concat[..., :self.kv_lora_rank],
                        q_nope)

        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # TODO: handle index / kv_cache correctly
        topk_indices_global = None

        # write the latent and rope to kv cache
        self.rope_with_kvcache(
            q_concat[..., self.kv_lora_rank:],
            None,
            q_pe,
            kv_lora,
            kv_cache,
            attn_metadata.slot_mapping.flatten(),
            positions,
            layer._k_scale,
        )
        q = q_concat

        if self.kv_cache_dtype != "fp8_ds_mla":
            attn_out = self._forward_bf16_kv(q, kv_cache, topk_indices_global,
                                             attn_metadata)
        else:
            attn_out = self._forward_fp8_kv(q, kv_cache, topk_indices_global,
                                            attn_metadata)

        self._v_up_proj(attn_out, out=output[:num_actual_toks])
        return output
