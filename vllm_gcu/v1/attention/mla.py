#!/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import vllm_gcu.kernels._custom_ops as ops

from vllm.attention.backends.abstract import AttentionType
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)

logger = init_logger(__name__)


class GCUMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["GCUMLAImpl"]:
        return GCUMLAImpl

    @staticmethod
    def get_metadata_cls() -> type["GCUMLAMetadata"]:
        return GCUMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["GCUMLAMetadataBuilder"]:
        return GCUMLAMetadataBuilder


@dataclass
class GCUMLADecodeMetadata(MLACommonDecodeMetadata):
    max_query_len: int


@dataclass
class GCUMLAMetadata(MLACommonMetadata[GCUMLADecodeMetadata]):
    pass


class GCUMLAMetadataBuilder(MLACommonMetadataBuilder[GCUMLAMetadata]):

    def _build_decode(
        self,
        input_positions: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> GCUMLADecodeMetadata:
        return GCUMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens,
            max_query_len=seq_lens.max().item(),
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

    def forward(
        self,
        layer,
        hidden_states_or_q_c,  # query in unified attn
        k_c_normed,  # key in unified attn
        k_pe,  # value in unified attn
        kv_cache,
        attn_metadata,
        output=None,
    ):
        # [TODO]: remove in v0.9.1
        if output is not None:
            output.fill_(0)

        return super().forward(layer, hidden_states_or_q_c, k_c_normed, k_pe, kv_cache, attn_metadata, output)

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

        decode_meta = attn_metadata.decode
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

        return self._v_up_proj_and_o_proj(o)
