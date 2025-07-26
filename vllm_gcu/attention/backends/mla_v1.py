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

    def _forward_decode(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: GCUMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        B = ql_nope.shape[0]

        q = torch.cat([ql_nope, q_pe], dim=-1)
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
