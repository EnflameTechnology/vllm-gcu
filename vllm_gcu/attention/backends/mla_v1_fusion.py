#!/usr/bin/env python
# coding=utf-8

from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.attention.backends.abstract import AttentionLayer
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
import vllm_gcu.kernels._custom_ops as ops
from vllm_gcu.attention.backends.mla_v1 import GCUMLABackend, GCUMLAImpl, GCUMLAMetadata
from vllm_gcu.attention.backends.mla_fusion import RopeWithKVCache

logger = init_logger(__name__)


class GCUMLAFusionBackend(GCUMLABackend):

    @staticmethod
    def get_impl_cls() -> type["GCUMLAImpl"]:
        return GCUMLAFusionImpl


class GCUMLAFusionImpl(GCUMLAImpl):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            rotary_emb: Optional[RotaryEmbedding] = None,
            kv_a_layernorm: Optional[torch.nn.Module] = None,
            # MLA Specific Arguments
            **mla_args) -> None:
        from flash_attn.vllm_flash_attn import flash_attn_varlen_func

        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)
        self.rotary_emb = rotary_emb
        self.kv_a_layernorm = kv_a_layernorm

        self.rope_with_kvcache = RopeWithKVCache(self.rotary_emb,
                                                 self.kv_a_layernorm,
                                                 self.kv_lora_rank,
                                                 self.qk_rope_head_dim,
                                                 kv_cache_dtype)

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self._pad_v = False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        self.W_UV = self.W_UV.contiguous()
        self.W_UK_T = self.W_UK_T.contiguous()

    def _v_up_proj(self, x):
        B = x.shape[0]
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
        # Multiply (B, N, L) x (N, L, V) -> (B, N, V)
        out = torch.empty((B, self.num_heads, self.W_UV.shape[-1]),
                          device=x.device,
                          dtype=x.dtype)
        torch.bmm(x.transpose(0, 1), self.W_UV, out=out.transpose(0, 1))
        # Convert from (B, N, V) to (B, N * V)
        return out.view(-1, self.num_heads * self.v_head_dim)

    def _k_up_proj(self, out, q_nope):
        B, N, P = q_nope.shape
        q_nope = q_nope
        # Multiply (B, N, P) x (N, P, L) -> (B, N, L)
        torch.bmm(q_nope.transpose(0, 1), self.W_UK_T, out=out.transpose(0, 1))

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,
        kv_c_and_k_pe: torch.Tensor,  # key in unified attn
        input_positions: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: GCUMLAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
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
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = hidden_states_or_q_c.view(-1, self.num_heads, self.qk_head_dim)
        q = q[:num_actual_toks, ...]
        kv_c_and_k_pe = kv_c_and_k_pe[:num_actual_toks, ...]
        input_positions = input_positions[:num_actual_toks, ...]

        assert attn_metadata.num_decodes is not None and \
            attn_metadata.num_prefills is not None and \
            attn_metadata.num_decode_tokens is not None

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]
        prefill_q = q[num_decode_tokens:]

        decode_input_positions = input_positions[:num_decode_tokens]
        prefill_input_positions = input_positions[num_decode_tokens:]
        decode_kv_c_and_k_pe = kv_c_and_k_pe[:num_decode_tokens]
        prefill_kv_c_and_k_pe = kv_c_and_k_pe[num_decode_tokens:]
        decode_slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
        prefill_slot_mapping = attn_metadata.slot_mapping[num_decode_tokens:]

        # write the latent and rope to kv cache
        if has_prefill:
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            # dim=-1: [v,k_nope,k_pe]
            # k_v_prefill = torch.empty((num_prefill_tokens, self.num_heads, self.qk_head_dim + self.v_head_dim))
            # prefill_k_pe = k_v_prefill[:, :, self.v_head_dim+self.qk_nope_head_dim]
            prefill_k_pe = torch.empty(
                (num_actual_toks - num_decode_tokens, self.num_heads,
                 self.qk_rope_head_dim),
                dtype=kv_c_and_k_pe.dtype,
                device=kv_c_and_k_pe.device,
            )
            prefill_k_c_normed = self.rope_with_kvcache(
                prefill_q_pe, prefill_k_pe,
                prefill_q_pe, prefill_kv_c_and_k_pe, kv_cache,
                prefill_slot_mapping.flatten(), prefill_input_positions,
                layer._k_scale)

        if has_decode:
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            decode_q_concat = torch.empty(
                (decode_q.shape[0], self.num_heads,
                 self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=decode_q.dtype,
                device=decode_q.device,
            )
            self._k_up_proj(decode_q_concat[..., :self.kv_lora_rank],
                            decode_q_nope)
            self.rope_with_kvcache(
                decode_q_concat[..., self.kv_lora_rank:],
                None,
                decode_q_pe,
                decode_kv_c_and_k_pe,
                kv_cache,
                decode_slot_mapping.flatten(),
                decode_input_positions,
                layer._k_scale,
            )
        if has_prefill:
            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata)

        if has_decode:
            output[:num_decode_tokens] = self._forward_decode(
                decode_q_concat, kv_cache, attn_metadata)

        return output_padded

    def _forward_decode(
        self,
        decode_q_concat: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: GCUMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        B = decode_q_concat.shape[0]

        q = decode_q_concat
        o = torch.zeros(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        ops.paged_attention_v1(
            out=o,
            query=q,
            key_cache=kv_c_and_k_pe_cache,
            value_cache=None,
            num_kv_heads=1,
            scale=self.scale,
            block_tables=decode_meta.block_table,
            seq_lens=decode_meta.seq_lens,
            block_size=kv_c_and_k_pe_cache.size(1),
            max_seq_len=decode_meta.max_query_len,
            alibi_slopes=None,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale_float=1.0,
            v_scale_float=1.0,
            out_scales=None,
        )

        return self._v_up_proj(o)
