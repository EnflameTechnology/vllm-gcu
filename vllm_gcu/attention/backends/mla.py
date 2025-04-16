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

import vllm_gcu.kernels._custom_ops as ops


class GCUMLABackend(MLACommonBackend):
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
    ):
        if attn_metadata is None:
            return torch.empty(
                [0, self.o_proj.output_size],
                dtype=hidden_states_or_q_c.dtype,
                device=hidden_states_or_q_c.device,
            )

        return super().forward(
            layer,
            hidden_states_or_q_c,
            k_c_normed,
            k_pe,
            kv_cache,
            attn_metadata,
            output,
        )

    def _forward_prefill(
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

        return self._v_up_proj_and_o_proj(o)
