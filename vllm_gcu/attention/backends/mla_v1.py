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
import vllm_gcu._C  # noqa
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
        return GCUMLADecodeMetadata(block_table=block_table_tensor,
                                    seq_lens=seq_lens,
                                    max_query_len=seq_lens.max().item())


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

    def _k_up_proj(self, out, q_nope):
        B, N, P = q_nope.shape
        q_nope = q_nope
        # Multiply (B, N, P) x (N, P, L) -> (B, N, L)
        torch.bmm(q_nope.transpose(0, 1), self.W_UK_T, out=out.transpose(0, 1))

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

        self._forward_decode = partial(self._forward_decode,
                                       k_scale=layer._k_scale_float)
        self._compute_prefill_context = partial(
            self._compute_prefill_context_gcu, k_scale=layer._k_scale)

        res = super().forward(layer, q, k_c_normed, k_pe, kv_cache,
                              attn_metadata, output, output_scale)
        if output is not None:
            return output.copy_(res)
        else:
            return res

    def _compute_prefill_context_gcu(self, q: torch.Tensor,
                                     kv_c_and_k_pe_cache: torch.Tensor,
                                     attn_metadata: MLACommonMetadata,
                                     k_scale: torch.Tensor):
        if attn_metadata.prefill.max_query_len < 8:
            context_output, context_lse = self._compute_prefill_context_flashmla(
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale)
            return context_output, context_lse
        else:
            return self._compute_prefill_context_ori(q, kv_c_and_k_pe_cache, attn_metadata)

    def _compute_prefill_context_flashmla(
        self,
        q: torch.
        Tensor,  # [num_prefill_tokens, num_heads, qk_head_dim(qk_nope_head_dim + qk_rope_head_dim)]
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: Optional[torch.Tensor] = None,
    ):
        from vllm_gcu.attention.ops.flashmla import flash_mla_with_kvcache

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        last_seq_starts = prefill_metadata.chunked_context.starts[-1]
        last_seq_lens = prefill_metadata.chunked_context.cu_seq_lens[-1].diff()
        cache_seqlens = last_seq_starts + last_seq_lens

        prefill_q_nope, prefill_q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        chunked_q = torch.empty(
            (q.shape[0], self.num_heads,
             self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=q.dtype,
            device=q.device)  # [num_tokens, num_heads, head_size]
        self._k_up_proj(chunked_q[..., :self.kv_lora_rank], prefill_q_nope)
        chunked_q[..., self.kv_lora_rank:].copy_(prefill_q_pe)

        num_prefills = attn_metadata.num_prefills
        max_query_len = prefill_metadata.max_query_len
        if num_prefills * max_query_len == chunked_q.shape[0]:
            chunked_q = chunked_q.reshape(num_prefills, max_query_len, *chunked_q.shape[1:])
            block_table = prefill_metadata.block_table
        else:
            chunked_q = chunked_q.unsqueeze(1)
            query_start_loc = prefill_metadata.query_start_loc
            query_lens =  query_start_loc[1:] - query_start_loc[:-1]
            repeat_indices = torch.repeat_interleave(
                torch.arange(num_prefills, device=query_lens.device),
                query_lens
            )
            block_table = prefill_metadata.block_table[repeat_indices]
            cache_seqlens = cache_seqlens[repeat_indices]

        q_scale = None
        if self.kv_cache_dtype == "fp8":
            assert k_scale is not None
            chunked_q, q_scale = ops.scaled_fp8_quant(chunked_q,
                                              q_scale,
                                              scale_ub=None,
                                              use_per_token_if_dynamic=True)


        attn_output, attn_softmax_lse = flash_mla_with_kvcache(
            q=chunked_q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=None,
            num_splits=None,
            softmax_scale=self.scale,
            causal=False,
            descale_q=q_scale,
            descale_k=k_scale,
        )

        attn_output = self._v_up_proj(
            attn_output.view(-1, *attn_output.shape[2:])).view(
                (-1, self.num_heads, self.v_head_dim
                 ))  # [num_mtp_prefill_tokens, num_heads, head_size_v]
        attn_softmax_lse = attn_softmax_lse.transpose(0, 1).view(
            self.num_heads, -1
        )  # [num_mtp_prefills, num_heads, query_len] -> [num_heads, num_mtp_prefill_tokens]

        return attn_output, attn_softmax_lse

    def _compute_prefill_context_ori(
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
        if self.kv_cache_dtype == "fp8":
            workspace = torch.empty_like(workspace, dtype=torch.float8_e4m3fn)

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

            workspace_gathered = workspace[:toks]
            if prefill_metadata.chunked_context.workspace.dtype != kv_c_and_k_pe_cache.dtype:
                workspace_gathered = workspace_gathered.to(prefill_metadata.chunked_context.workspace.dtype)

            kv_c_normed = workspace_gathered[..., :self.kv_lora_rank]
            k_pe = workspace_gathered[..., self.kv_lora_rank:].unsqueeze(1)

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

    def _forward_decode(self, ql_nope: torch.Tensor, q_pe: torch.Tensor,
                        kv_c_and_k_pe_cache: torch.Tensor,
                        attn_metadata: GCUMLAMetadata,
                        k_scale: float) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        # if self.kv_cache_dtype.startswith("fp8"):
        #     raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        B = ql_nope.shape[0]

        q = torch.cat([ql_nope, q_pe], dim=-1)
        o = torch.empty(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        q_scale = None
        if self.kv_cache_dtype == "fp8":
            q, q_scale = ops.scaled_fp8_quant(q,
                                              q_scale,
                                              scale_ub=None,
                                              use_per_token_if_dynamic=True)

        ops.paged_attention_v1(out=o,
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
                               query_scales=q_scale)

        return self._v_up_proj(o)
