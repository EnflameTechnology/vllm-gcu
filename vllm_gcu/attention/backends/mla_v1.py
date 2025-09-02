#!/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)

import vllm_gcu.kernels._custom_ops as ops
import vllm_gcu._C
from vllm_gcu.kernels._custom_ops import merge_attn_states

logger = init_logger(__name__)


class GCUMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["GCUMLAMetadata"]:
        return GCUMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["GCUMLAMetadataBuilder"]:
        return GCUMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["GCUMLAImpl"]:
        return GCUMLAImpl


@dataclass
class GCUMLADecodeMetadata(MLACommonDecodeMetadata):
    max_query_len: int


@dataclass
class GCUMLAMetadata(MLACommonMetadata[GCUMLADecodeMetadata]):
    pass


class GCUMLAMetadataBuilder(MLACommonMetadataBuilder[GCUMLAMetadata]):
    full_cudagraph_supported = True

    def build_for_cudagraph_capture(self, common_attn_metadata):
        m = common_attn_metadata
        m.max_query_len = 1

        self._num_decodes = m.num_reqs
        self._num_decode_tokens = m.num_actual_tokens
        self._num_prefills = 0
        self._num_prefill_tokens = 0
        return self.build(0, m)

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens: torch.Tensor):
        return GCUMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens,
            max_query_len=seq_lens.max().item()
        )


class GCUMLAImpl(MLACommonImpl[GCUMLAMetadata]):
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
            # MLA Specific Arguments
            **mla_args) -> None:
        from flash_attn.vllm_flash_attn import flash_attn_varlen_func

        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self._pad_v = False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        self.W_UV = self.W_UV.contiguous()
        self.W_UK_T = self.W_UK_T.contiguous()

    def forward(
        self,
        layer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        from functools import partial
        self._forward_decode = partial(self._forward_decode, k_scale=layer._k_scale_float)
        res = super().forward(
            layer,
            q,
            k_c_normed,
            k_pe,
            kv_cache,
            attn_metadata,
            output,
            output_scale
        )
        if output is not None:
            return output.copy_(res)
        else:
            return res

    def _forward_decode(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: GCUMLAMetadata,
        k_scale: float
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        # if self.kv_cache_dtype.startswith("fp8"):
        #     raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        B = ql_nope.shape[0]

        q = torch.cat([ql_nope, q_pe], dim=-1)
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
            block_tables=decode_meta.block_table,
            seq_lens=decode_meta.seq_lens,
            block_size=kv_c_and_k_pe_cache.size(1),
            max_seq_len=decode_meta.max_query_len,
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
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace
        if prefill_metadata.chunked_context.workspace.dtype != kv_c_and_k_pe_cache.dtype:
            workspace = prefill_metadata.chunked_context.workspace.to(kv_c_and_k_pe_cache.dtype)
        else:
            workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]

            torch.ops._C_cache_ops.gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )

            if prefill_metadata.chunked_context.workspace.dtype != kv_c_and_k_pe_cache.dtype:
                workspace = workspace.to(prefill_metadata.chunked_context.workspace.dtype)

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
                cu_seqlens_k=prefill_metadata.chunked_context.cu_seq_lens[i],
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=prefill_metadata.chunked_context.max_seq_lens[i],
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
