#!/usr/bin/env python
# coding=utf-8
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.attention.backends.mla.common import (
    MLACommonMetadata,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

import vllm_gcu.kernels._custom_ops as ops
from vllm_gcu.attention.backends.mla import GCUMLABackend, GCUMLAImpl


class GCUMLAFusionBackend(GCUMLABackend):

    @staticmethod
    def get_impl_cls() -> Type["GCUMLAFusionImpl"]:
        return GCUMLAFusionImpl


@CustomOp.register("rope_with_kvcache")
class RopeWithKVCache(CustomOp):
    cos_sin_cache = None

    def __init__(self, rotary_emb, kv_a_layernorm, kv_lora_rank,
                 qk_rope_head_dim, kv_cache_dtype):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_dtype = kv_cache_dtype
        if RopeWithKVCache.cos_sin_cache is None:
            RopeWithKVCache.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(
                current_platform.device_type,
                dtype=torch.float32,
            )

    def forward(
        self,
        q_pe_out,
        k_pe_out,
        q_pe,
        kv_c_and_k_pe,
        kv_cache,
        slot_mapping,
        input_positions,
        kv_scale,
        k_c_normed_out=None,
    ):
        dispatch = super().forward
        prefill_support_platform = [140]
        if (current_platform.get_device_capability().to_int() not in prefill_support_platform \
                and k_pe_out is not None) or kv_cache.numel() == 0:
            # prefill use native impl since op interface lack outputs.
            dispatch = self.forward_native
        return dispatch(
            q_pe_out,
            k_pe_out,
            q_pe,
            kv_c_and_k_pe,
            kv_cache,
            slot_mapping,
            input_positions,
            kv_scale,
            k_c_normed_out,
        )

    def forward_native(
        self,
        q_pe_out,
        k_pe_out,
        q_pe,
        kv_c_and_k_pe,
        kv_cache,
        slot_mapping,
        input_positions,
        kv_scale,
        k_c_normed_out=None,
    ):
        kv_c, k_pe = kv_c_and_k_pe.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_c_normed = self.kv_a_layernorm(kv_c)
        if k_c_normed_out is not None:
            k_c_normed_out.copy_(k_c_normed)
        k_pe = k_pe.unsqueeze(1)

        q_pe_out[...], k_pe[...] = self.rotary_emb(input_positions, q_pe, k_pe)
        if k_pe_out is not None:
            k_pe_out[...] = k_pe

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            from vllm import _custom_ops as vops
            vops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                slot_mapping,
                kv_cache_dtype=self.kv_cache_dtype,
                scale=kv_scale,
            )

    def forward_oot(
        self,
        q_pe_out,
        k_pe_out,
        q_pe,
        kv_c_and_k_pe,
        kv_cache,
        slot_mapping,
        input_positions,
        kv_scale,
        k_c_normed_out=None,
    ):
        torch.ops._C.rotary_embedding_with_kv_cache(
            q_pe_out, kv_cache, k_pe_out, k_c_normed_out, q_pe, kv_c_and_k_pe,
            input_positions, RopeWithKVCache.cos_sin_cache,
            self.kv_a_layernorm.weight.data, slot_mapping, kv_scale,
            self.kv_a_layernorm.variance_epsilon,
            [self.kv_lora_rank, self.qk_rope_head_dim], self.kv_cache_dtype)



class GCUMLAFusionImpl(GCUMLAImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]],
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        kv_a_layernorm: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            blocksparse_params,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        self.rotary_emb = rotary_emb
        self.kv_a_layernorm = kv_a_layernorm

        self.rope_with_kvcache = RopeWithKVCache(
            self.rotary_emb, self.kv_a_layernorm, self.kv_lora_rank, self.qk_rope_head_dim, kv_cache_dtype)

    def _v_up_proj(self, x):
        B = x.shape[0]
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
        # Multiply (B, N, L) x (N, L, V) -> (B, N, V)
        out = torch.empty((B, self.num_heads, self.W_UV.shape[-1]), device=x.device, dtype=x.dtype)
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
        layer,
        hidden_states_or_q_c,
        kv_c_and_k_pe,
        input_positions,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
    ):
        if attn_metadata is None:
            if output is not None:
                return output
            else:
                return torch.empty_like(hidden_states_or_q_c)

        if attn_metadata.is_profile_run and \
                attn_metadata.context_chunk_workspace is not None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (attn_metadata.context_chunk_workspace.shape[0],
                 self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
                device=kv_c_and_k_pe.device,
                dtype=kv_c_and_k_pe.dtype,
            )

        has_decode = attn_metadata.decode_metadata is not None
        has_prefill = attn_metadata.prefill_metadata is not None
        num_prefill_tokens: int = attn_metadata.num_prefill_tokens

        q = hidden_states_or_q_c.view(-1, self.num_heads, self.qk_head_dim)
        decode_q = q[num_prefill_tokens:]
        prefill_q = q[:num_prefill_tokens]
        decode_input_positions = input_positions[num_prefill_tokens:]
        prefill_input_positions = input_positions[:num_prefill_tokens]
        decode_kv_c_and_k_pe = kv_c_and_k_pe[num_prefill_tokens:]
        prefill_kv_c_and_k_pe = kv_c_and_k_pe[:num_prefill_tokens]
        decode_slot_mapping = attn_metadata.slot_mapping[num_prefill_tokens:]
        prefill_slot_mapping = attn_metadata.slot_mapping[:num_prefill_tokens]

        if has_prefill:
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            # dim=-1: [v,k_nope,k_pe]
            # k_v_prefill = torch.empty((num_prefill_tokens, self.num_heads, self.qk_head_dim + self.v_head_dim))
            # prefill_k_pe = k_v_prefill[:, :, self.v_head_dim+self.qk_nope_head_dim]
            prefill_k_pe = torch.empty(
                (num_prefill_tokens, 1, self.qk_rope_head_dim),
                dtype=kv_c_and_k_pe.dtype,
                device=kv_c_and_k_pe.device,
            )
            prefill_k_c_normed = torch.empty(
                (num_prefill_tokens, self.kv_lora_rank),
                dtype=kv_c_and_k_pe.dtype,
                device=kv_c_and_k_pe.device,
            )
            self.rope_with_kvcache(
                prefill_q_pe,
                prefill_k_pe,
                prefill_q_pe,
                prefill_kv_c_and_k_pe,
                kv_cache,
                prefill_slot_mapping.flatten(),
                prefill_input_positions,
                layer._k_scale,
                prefill_k_c_normed,
            )

        if has_decode:
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            decode_q_concat = torch.empty(
                (decode_q.shape[0], self.num_heads, self.kv_lora_rank+self.qk_rope_head_dim),
                dtype=decode_q.dtype,
                device=decode_q.device,
            )
            self._k_up_proj(decode_q_concat[...,:self.kv_lora_rank],decode_q_nope)
            self.rope_with_kvcache(
                decode_q_concat[...,self.kv_lora_rank:],
                None,
                decode_q_pe,
                decode_kv_c_and_k_pe,
                kv_cache,
                decode_slot_mapping.flatten(),
                decode_input_positions,
                layer._k_scale,
            )

        if output is None:
            output = torch.empty(attn_metadata.num_prefill_tokens +
                                 attn_metadata.num_decode_tokens,
                                 self.o_proj.output_size,
                                 device=hidden_states_or_q_c.device,
                                 dtype=hidden_states_or_q_c.dtype)
        if has_prefill:
            output[:num_prefill_tokens] = self._forward_prefill(
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata)

        if has_decode:
            output[num_prefill_tokens:] = self._forward_decode(
                decode_q_concat, kv_cache, attn_metadata, layer._k_scale_float)

        return output

    def _forward_decode(
        self,
        decode_q_concat: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: float
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        # if self.kv_cache_dtype.startswith("fp8"):
        #     raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None
        B = decode_q_concat.shape[0]

        q = decode_q_concat
        o = torch.empty(
            B, self.num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        q_scale=None
        if self.kv_cache_dtype=="fp8":
            q, q_scale = ops.scaled_fp8_quant(q, q_scale, scale_ub=None, use_per_token_if_dynamic=True)
            
        ops.paged_attention_v1(
            out=o,
            query=q,
            key_cache=kv_c_and_k_pe_cache,
            value_cache=None,
            num_kv_heads=1,
            scale=self.scale,
            block_tables=decode_meta.block_tables,
            seq_lens=decode_meta.seq_lens_tensor,
            block_size=kv_c_and_k_pe_cache.size(1),
            max_seq_len=decode_meta.max_decode_seq_len,
            alibi_slopes=None,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale_float=k_scale,
            v_scale_float=k_scale,
            out_scales=None,
            query_scales=q_scale
        )

        return self._v_up_proj(o)
