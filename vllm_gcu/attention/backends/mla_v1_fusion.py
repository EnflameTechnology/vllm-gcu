#!/usr/bin/env python
# coding=utf-8

from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.attention.backends.abstract import AttentionLayer
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.distributed.parallel_state import get_dcp_group
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm_gcu.attention.backends.mla_v1 import GCUMLABackend, GCUMLAImpl, GCUMLAMetadata


logger = init_logger(__name__)


class GCUMLAFusionBackend(GCUMLABackend):

    @staticmethod
    def get_impl_cls() -> type["GCUMLAImpl"]:
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
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
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
                         logits_soft_cap, attn_type,
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

    def _v_up_proj(self, x, out=None):
        B = x.shape[0]
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
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

    def _k_up_proj(self, out, q_nope):
        B, N, P = q_nope.shape
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
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

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

        if self.dcp_world_size is None:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        if fp8_attention:
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

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
                (num_actual_toks - num_decode_tokens, 1,
                 self.qk_rope_head_dim),
                dtype=kv_c_and_k_pe.dtype,
                device=kv_c_and_k_pe.device,
            )
            prefill_k_c_normed = torch.empty(
                (num_actual_toks - num_decode_tokens, self.kv_lora_rank),
                dtype=kv_c_and_k_pe.dtype,
                device=kv_c_and_k_pe.device,
            )
            self.rope_with_kvcache(prefill_q_pe, prefill_k_pe, prefill_q_pe,
                                   prefill_kv_c_and_k_pe, kv_cache,
                                   prefill_slot_mapping.flatten(),
                                   prefill_input_positions, layer._k_scale,
                                   prefill_k_c_normed)

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
                attn_metadata, layer._k_scale)

        if has_decode:
            if self.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # decode_q do allgather in head dim.
                decode_q_concat = get_dcp_group().all_gather(decode_q_concat,
                                                             dim=1)
            attn_out, lse = self._forward_decode(decode_q_concat, kv_cache,
                                                 attn_metadata, layer)

            # recorect dcp attn_out with lse.
            if self.dcp_world_size > 1:
                attn_out = cp_lse_ag_out_rs(attn_out, lse, get_dcp_group())

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])

        return output_padded
