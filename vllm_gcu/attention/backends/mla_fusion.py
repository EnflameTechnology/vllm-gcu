#!/usr/bin/env python
# coding=utf-8
import itertools
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

from vllm import envs
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
import vllm_gcu.kernels._custom_ops as ops
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    is_vllm_fa = True
except ImportError:
    # For rocm use upstream flash attention
    from flash_attn import flash_attn_varlen_func
    is_vllm_fa = False

is_hip = current_platform.is_rocm()


class GCUMLABackend(MLACommonBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> Type["GCUMLAImpl"]:
        return GCUMLAImpl

    @staticmethod
    def get_builder_cls() -> Type["MLACommonMetadataBuilder"]:
        return GCUMLACommonMetadataBuilder

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        # ops.copy_blocks_mla(kv_caches, src_to_dists)
        raise NotImplementedError


class GCUMLACommonMetadataBuilder(MLACommonMetadataBuilder):
    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        cuda_graph_pad_size: int,
        batch_size: int,
    ):
        use_captured_graph = cuda_graph_pad_size != -1
        if use_captured_graph:
            self.input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
        return super().build(seq_lens, query_lens, cuda_graph_pad_size, batch_size)


@CustomOp.register("rope_with_kvcache")
class RopeWithKVCache(CustomOp):
    def __init__(self, rotary_emb, kv_a_layernorm, kv_lora_rank, qk_rope_head_dim, kv_cache_dtype):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_dtype = kv_cache_dtype

    def forward(self,
                q_pe_out,
                k_pe_out,
                q_pe,
                kv_c_and_k_pe,
                kv_cache,
                slot_mapping,
                input_positions,
                kv_scale,):
        dispatch = super().forward
        support_platform = [130, 140]
        if current_platform.get_device_capability().to_int() not in support_platform \
                or k_pe_out is not None:
            # prefill use native impl since op interface lack outputs.
            dispatch = self.forward_native
        return dispatch(q_pe_out,
                        k_pe_out,
                        q_pe,
                        kv_c_and_k_pe,
                        kv_cache,
                        slot_mapping,
                        input_positions,
                        kv_scale,)

    def forward_native(self,
                       q_pe_out,
                       k_pe_out,
                       q_pe,
                       kv_c_and_k_pe,
                       kv_cache,
                       slot_mapping,
                       input_positions,
                       kv_scale,
                       ):
        kv_c, k_pe = kv_c_and_k_pe.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_c_normed = self.kv_a_layernorm(kv_c.contiguous())
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
        if k_pe_out is not None:
            return k_c_normed

    def forward_oot(self,
                    q_pe_out,
                    k_pe_out,
                    q_pe,
                    kv_c_and_k_pe,
                    kv_cache,
                    slot_mapping,
                    input_positions,
                    kv_scale,
                    ):
        if self.rotary_emb.cos_sin_cache.device != q_pe.device or \
                self.rotary_emb.cos_sin_cache.dtype != torch.float32:
            self.rotary_emb.cos_sin_cache = \
                self.rotary_emb.cos_sin_cache.to(q_pe.device, dtype=torch.float32)
        torch.ops._C.rotary_embedding_with_kv_cache(
            q_pe_out, kv_cache, q_pe, kv_c_and_k_pe, input_positions,
            self.rotary_emb.cos_sin_cache, self.kv_a_layernorm.weight.data,
            slot_mapping, kv_scale, self.kv_a_layernorm.variance_epsilon,
            [self.kv_lora_rank, self.qk_rope_head_dim], self.kv_cache_dtype
        )


class GCUMLAImpl(MLACommonImpl[MLACommonMetadata]):

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
        kv_a_layernorm: Optional[torch.nn.Module],
        **kwargs,
    ) -> None:
        from flash_attn.vllm_flash_attn import flash_attn_varlen_func

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
            **kwargs,
        )
        self.kv_a_layernorm = kv_a_layernorm

        unsupported_features = [
            alibi_slopes,
            sliding_window,
            blocksparse_params,
            logits_soft_cap,
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "GCUMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention are not implemented for GCUMLAImpl"
            )

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.rope_with_kvcache = RopeWithKVCache(
            self.rotary_emb, self.kv_a_layernorm, self.kv_lora_rank, self.qk_rope_head_dim, kv_cache_dtype)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        self.W_UV = self.W_UV.contiguous()
        self.W_UK_T = self.W_UK_T.contiguous()

    def _v_up_proj(self, x):
        B = x.shape[0]
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
        # Multiply (B, N, L) x (N, L, V) -> (B, N, V)
        out = torch.empty((B, self.num_heads, self.W_UV.shape[-1]), device=x.device, dtype=x.dtype)
        torch.bmm(x.transpose(0, 1), self.W_UV, out=out.transpose(0, 1))
        # Convert from (B, N, V) to (B, N * V)
        return out.view(-1, self.num_heads * self.v_head_dim)

    def _k_up_proj(self, q_nope):
        B, N, P = q_nope.shape
        q_nope = q_nope
        # Multiply (B, N, P) x (N, P, L) -> (B, N, L)
        ql_nope = torch.empty((B, N, self.W_UK_T.shape[-1]), device=q_nope.device, dtype=q_nope.dtype)
        torch.bmm(q_nope.transpose(0, 1), self.W_UK_T, out=ql_nope.transpose(0, 1))
        return ql_nope

    def forward(
        self,
        layer,
        hidden_states_or_q_c,
        kv_c_and_k_pe,
        place_holder,
        kv_cache,
        attn_metadata,
        output=None,
    ):
        if attn_metadata is None:
            if output is not None:
                return output
            else:
                return torch.empty(
                    [0, self.o_proj.output_size],
                    dtype=hidden_states_or_q_c.dtype,
                    device=hidden_states_or_q_c.device,
                )

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
        assert hasattr(attn_metadata, "input_positions")
        num_prefill_tokens: int = attn_metadata.num_prefill_tokens

        q = self.q_proj(hidden_states_or_q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
        decode_q = q[num_prefill_tokens:]
        prefill_q = q[:num_prefill_tokens]
        decode_input_positions = \
            attn_metadata.input_positions[num_prefill_tokens:]
        prefill_input_positions = \
            attn_metadata.input_positions[:num_prefill_tokens]

        if has_prefill:
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            # dim=-1: [v,k_nope,k_pe]
            # k_v_prefill = torch.empty((num_prefill_tokens, self.num_heads, self.qk_head_dim + self.v_head_dim))
            # prefill_k_pe = k_v_prefill[:, :, self.v_head_dim+self.qk_nope_head_dim]
            prefill_k_pe = torch.empty(
                (num_prefill_tokens, self.num_heads, self.qk_rope_head_dim),
                dtype=kv_c_and_k_pe.dtype,
                device=kv_c_and_k_pe.device,
            )
            prefill_k_c_normed = self.rope_with_kvcache(
                prefill_q_pe,
                prefill_k_pe,
                prefill_q_pe,
                kv_c_and_k_pe,
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                prefill_input_positions,
                layer._k_scale
            )

        if has_decode:
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            decode_ql_nope = self._k_up_proj(decode_q_nope)
            self.rope_with_kvcache(
                decode_q_pe,
                None,
                decode_q_pe,
                kv_c_and_k_pe,
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
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
                decode_ql_nope, decode_q_pe, kv_cache, attn_metadata)

        return output

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        if current_platform.get_device_capability()[0] == 13:
            return self._forward_prefill_xformers(
                q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata
            )

        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None

        has_context = prefill_metadata.context_lens_tensor is not None \
            and prefill_metadata.context_lens_tensor.max() > 0

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        if is_hip and envs.VLLM_USE_TRITON_FLASH_ATTN and not has_context:
            output = self.triton_fa_func(
                q,
                k,
                v_padded,
                None,
                prefill_metadata.query_start_loc,
                prefill_metadata.query_start_loc,
                prefill_metadata.max_prefill_seq_len,
                prefill_metadata.max_prefill_seq_len,
                True,  # causal
                self.scale,
                None,  # attn_mask is None unless applying ALiBi mask
            )
            # triton flash attention always return 2 objects
            if not has_context:
                output = output[0]
        elif is_vllm_fa:
            output = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v_padded,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.query_start_loc,
                max_seqlen_q=prefill_metadata.max_prefill_seq_len,
                max_seqlen_k=prefill_metadata.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                return_softmax_lse=has_context,
            )
        else:
            output = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v_padded,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.query_start_loc,
                max_seqlen_q=prefill_metadata.max_prefill_seq_len,
                max_seqlen_k=prefill_metadata.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                return_attn_probs=has_context,
            )

        if has_context:
            # ROCm flash_attn_varlen_func will return 3 objects instead of 2
            suffix_output, suffix_lse, *rest = output
            context_output, context_lse = self._compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # slice by `:v.shape[-1]` in order to remove v headdim padding
        output = output\
            .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
            .reshape(-1, self.num_heads * v.shape[-1])

        return self.o_proj(output)[0]

    def _forward_prefill_xformers(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, MLACommonMetadata)

        prefill_metadata = attn_metadata.prefill_metadata
        if (
            prefill_metadata.context_lens_tensor is not None
            and prefill_metadata.context_lens_tensor.max() > 0
        ):
            raise NotImplementedError

        from xformers import ops as xops
        from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

        original_query = q

        attn_bias = BlockDiagonalCausalMask.from_seqlens(
            attn_metadata.seq_lens, device="gcu"
        )

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)

        attn_output = xops.memory_efficient_attention_forward(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v_padded.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scale,
        )
        attn_output = attn_output.view(-1, self.num_heads, original_query.shape[-1])[
            ..., : v.shape[-1]
        ].reshape(-1, self.num_heads * v.shape[-1])
        return self.o_proj(attn_output)[0]

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None
        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(
            B, self.num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

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
            k_scale_float=1.0,
            v_scale_float=1.0,
            out_scales=None,
        )

        x = self._v_up_proj(o)
        return self.o_proj(x)[0]
