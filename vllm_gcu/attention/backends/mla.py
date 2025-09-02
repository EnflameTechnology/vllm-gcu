#!/usr/bin/env python
# coding=utf-8
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING
from unittest.mock import patch

import torch
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)

from vllm.platforms import current_platform


import vllm_gcu.kernels._custom_ops as ops
import vllm_gcu._C
from vllm_gcu.kernels._custom_ops import merge_attn_states

if TYPE_CHECKING:
    pass


class GCUMLABackend(MLACommonBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> Type["GCUMLAImpl"]:
        return GCUMLAImpl

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
        kv_sharing_target_layer_name: Optional[str] = None,
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

        self._pad_v = False  # only for flash attn
        try:
            from flash_attn.vllm_flash_attn import flash_attn_varlen_func
            self.flash_attn_varlen_func = flash_attn_varlen_func
        except Exception:
            self.flash_attn_varlen_func = None

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        self.W_UV = self.W_UV.contiguous()
        self.W_UK_T = self.W_UK_T.contiguous()

    def forward(
        self,
        layer,
        hidden_states_or_q_c,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
    ):
        if attn_metadata is None:
            if output is not None:
                return output
            else:
                return torch.empty_like(hidden_states_or_q_c).contiguous()

        # if self.kv_cache_dtype.startswith("fp8"): 
        from functools import partial
        self._forward_decode = partial(self._forward_decode, k_scale=layer._k_scale_float)
 
        res = super().forward(
            layer,
            hidden_states_or_q_c,
            k_c_normed,
            k_pe,
            kv_cache,
            attn_metadata,
            None,
        )

        if output is not None:
            return output.copy_(res)
        else:
            return res

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:

        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None

        has_context = (
            prefill_metadata.context_lens_tensor is not None
            and prefill_metadata.context_lens_tensor.max() > 0
        )

        if current_platform.get_device_capability()[0] == 13 and not has_context:
            return self._forward_prefill_xformers(
                q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata
            )

        with patch("vllm.attention.backends.mla.common.merge_attn_states", ops.merge_attn_states):
            return super()._forward_prefill(q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata)

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
            prefill_metadata.seq_lens, device="gcu"
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
        return attn_output

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: float
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        # if self.kv_cache_dtype.startswith("fp8"):
        #     raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None
        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(
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
    
    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ):
        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None
        assert prefill_metadata.context_chunk_seq_tot is not None
        assert prefill_metadata.context_chunk_cu_seq_lens is not None
        assert prefill_metadata.context_chunk_starts is not None
        assert prefill_metadata.context_chunk_max_seq_lens is not None
        assert prefill_metadata.context_lens_tensor is not None

        output = None
        iters = len(prefill_metadata.context_chunk_seq_tot)

        # Fetch from attn_metadata directly, since it late bound by
        # MLAAttentionState, grabbing it directly `attn_metadata` can avoid
        # any weirdness around prefill_metadata caching
        assert attn_metadata.context_chunk_workspace is not None
        if attn_metadata.context_chunk_workspace.dtype != kv_c_and_k_pe_cache.dtype:
            workspace = attn_metadata.context_chunk_workspace.to(kv_c_and_k_pe_cache.dtype)
        else:
            workspace = attn_metadata.context_chunk_workspace

        for i in range(iters):
            toks = prefill_metadata.context_chunk_seq_tot[i]

            torch.ops._C_cache_ops.gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_tables,
                cu_seq_lens=prefill_metadata.context_chunk_cu_seq_lens[i],
                batch_size=prefill_metadata.num_prefills,
                seq_starts=prefill_metadata.context_chunk_starts[i],
            )

            if attn_metadata.context_chunk_workspace.dtype != kv_c_and_k_pe_cache.dtype:
                workspace = workspace.to(attn_metadata.context_chunk_workspace.dtype)

            kv_c_normed = workspace[:toks]\
                [..., :self.kv_lora_rank]
            k_pe = workspace[:toks]\
                [..., self.kv_lora_rank:].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))),
                          dim=-1)

            attn_output, attn_softmax_lse = \
                self._flash_attn_varlen_diff_headdims(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.context_chunk_cu_seq_lens[i],
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=prefill_metadata.context_chunk_max_seq_lens[i],
                softmax_scale=self.scale,
                causal=False,  # Context is unmasked
                return_softmax_lse=True,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse
