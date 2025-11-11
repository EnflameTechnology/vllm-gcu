#!/usr/bin/env python
# coding=utf-8

from vllm.attention.backends.abstract import AttentionLayer
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend, MLACommonDecodeMetadata, MLACommonImpl,
    MLACommonMetadata, MLACommonMetadataBuilder, M, split_decodes_and_prefills)
from vllm.v1.attention.backends.utils import AttentionCGSupport, CommonAttentionMetadata

import torch
import vllm_gcu.kernels._custom_ops as gops
import vllm._custom_ops as ops
import vllm_gcu.envs as gcu_envs
from dataclasses import dataclass
from typing import Any, Optional, Union
from functools import partial
from unittest.mock import patch

from vllm_gcu.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.config import VllmConfig


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
    max_decode_seq_len: int
    tile_scheduler_metadata: torch.Tensor
    num_splits: torch.Tensor


@dataclass
class GCUMLAMetadata(MLACommonMetadata[GCUMLADecodeMetadata]):
    is_for_decode_gcu_graph: bool = False

def customized_split_decodes_and_prefills(
        common_attn_metadata: CommonAttentionMetadata,
        decode_threshold: int = 1,
        require_uniform: bool = False,
        builder = None) -> tuple[int, int, int, int]:
    if hasattr(builder, "_num_decodes") and builder._num_decodes is not None and \
        hasattr(builder, "_num_prefills") and builder._num_prefills is not None and \
        hasattr(builder, "_num_decode_tokens") and builder._num_decode_tokens is not None and \
        hasattr(builder, "_num_prefill_tokens") and builder._num_prefill_tokens is not None:
        return builder._num_decodes, builder._num_prefills, \
            builder._num_decode_tokens, builder._num_prefill_tokens
    return split_decodes_and_prefills(common_attn_metadata = common_attn_metadata,
                                        decode_threshold = decode_threshold,
                                        require_uniform = require_uniform)


class GCUMLAMetadataBuilder(MLACommonMetadataBuilder[GCUMLAMetadata]):
    reorder_batch_threshold: int = 8
    # NOTE: uniform decode graphs will only be selected when q=N*(1+k)
    cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         GCUMLAMetadata)

        self.use_tile_scheduler_metadata = True if \
          current_platform.get_device_capability().to_int() == 140 else False

        self.cg_buf_tile_scheduler_metadata = torch.empty(
            24 * 1024 * 1024,
            device=self.device,
            dtype=torch.int8,
        ) if self.use_tile_scheduler_metadata else None


    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> M:
        with patch(
                'vllm.v1.attention.backends.mla.common.split_decodes_and_prefills',
                partial(customized_split_decodes_and_prefills, builder = self, \
                         require_uniform=True)):
            return super().build(common_prefix_len, common_attn_metadata,
                                 fast_build)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        m = common_attn_metadata
        if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
            if m.num_actual_tokens > 0:
                assert m.num_actual_tokens % m.max_query_len == 0 and \
                    m.num_reqs == m.num_actual_tokens // m.max_query_len
            self._num_decodes = m.num_reqs
            self._num_decode_tokens = m.num_actual_tokens
            self._num_prefills = 0
            self._num_prefill_tokens = 0
            metadata = self.build(0, m)
            metadata.is_for_decode_gcu_graph = True
            # for fused_mtp we don't decode attention does not rely on
            # metadata.decode.max_decode_seq_len
        elif m.num_actual_tokens == 0:
            self._num_decodes = m.num_reqs
            m.max_query_len = 1
            self._num_decode_tokens = m.num_actual_tokens
            self._num_prefills = 0
            self._num_prefill_tokens = 0
            metadata = super().build_for_cudagraph_capture(m)
            # overwrite max_decode_seq_len to max_model_len when capture
            metadata.decode.max_decode_seq_len = m.max_seq_len
        else:
            metadata = super().build_for_cudagraph_capture(m)
            # overwrite max_decode_seq_len to max_model_len when capture
            metadata.decode.max_decode_seq_len = m.max_seq_len
        return metadata

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens_cpu: torch.Tensor,
                      seq_lens_device: torch.Tensor,
                      query_start_loc_cpu: torch.Tensor,
                      query_start_loc_device: torch.Tensor,
                      num_decode_tokens: int):
        # important if async-scheduling is enable later
        if hasattr(self, 'max_seq_len'):
            max_seq_len = self.max_seq_len
        elif seq_lens_cpu is not None:
            max_seq_len = seq_lens_cpu.max()
        else:
            max_seq_len = seq_lens_device.max().item()

        if self.use_tile_scheduler_metadata:
            get_mla_metadata(self.cg_buf_tile_scheduler_metadata, seq_lens_device)

        return GCUMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            max_decode_seq_len=max_seq_len,
            tile_scheduler_metadata=self.cg_buf_tile_scheduler_metadata,
            num_splits=None,
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
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        from flash_attn.vllm_flash_attn import flash_attn_varlen_func

        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self._pad_v = False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        self.W_UV = self.W_UV.contiguous()
        self.W_UK_T = self.W_UK_T.contiguous()

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
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
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        assert attn_metadata.num_decodes is not None and \
            attn_metadata.num_prefills is not None and \
            attn_metadata.num_decode_tokens is not None

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]

        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if has_prefill:
            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata, layer._k_scale)

        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            # NOTE: we use dynamic q quant when c8, which is in forward_decode

            # if fp8_attention:
            #     ql_nope_shape = decode_ql_nope.shape
            #     decode_ql_nope, _ = ops.scaled_fp8_quant(
            #         decode_ql_nope.reshape([
            #             ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]
            #         ]), layer._q_scale)
            #     decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
            #     q_pe_shape = decode_q_pe.shape
            #     decode_q_pe, _ = ops.scaled_fp8_quant(
            #         decode_q_pe.reshape(
            #             [q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
            #         layer._q_scale)
            #     decode_q_pe = decode_q_pe.reshape(q_pe_shape)

            decode_q = (decode_ql_nope, decode_q_pe)
            if self.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # concatenate decode_ql_nope and decode_q_pe -> (B, N, L + P)
                decode_q = torch.cat(decode_q, dim=-1)
                # decode_q do allgather in head dim.
                decode_q = get_dcp_group().all_gather(decode_q, dim=1)

            # call decode attn
            attn_out, lse = self._forward_decode(decode_q, kv_cache,
                                                 attn_metadata, layer)

            # recorect dcp attn_out with lse.
            if self.dcp_world_size > 1:
                attn_out = cp_lse_ag_out_rs(attn_out, lse, get_dcp_group())

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])
        return output_padded

    def _forward_decode(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: GCUMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        # if self.kv_cache_dtype.startswith("fp8"):
        #     raise NotImplementedError("FP8 MLA not yet supported")

        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        q_dtype = q.dtype

        q_scale = None
        if self.kv_cache_dtype == "fp8":
            q, q_scale = gops.scaled_fp8_quant(q,
                                               q_scale,
                                               scale_ub=None,
                                               use_per_token_if_dynamic=True)

        sum_seq_q = q.shape[0]
        batch = decode_meta.block_table.shape[0]

        if sum_seq_q // batch > 1:
            assert sum_seq_q % batch == 0
            q = q.view(batch, sum_seq_q // batch, *q.shape[1:])
            if q_scale is not None:
                q_scale = q_scale.view(batch, sum_seq_q // batch, *q_scale.shape[1:])
            output, softmax_lse = flash_mla_with_kvcache(
                q=q,
                k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),
                block_table=decode_meta.block_table,
                cache_seqlens=decode_meta.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=decode_meta.tile_scheduler_metadata,
                num_splits=decode_meta.num_splits,
                softmax_scale=self.scale,
                causal=True,
                descale_q=q_scale,
                descale_k=layer._k_scale,
            )
            output = output.view(-1, *output.shape[2:])
            # TODO: for dcp
            # softmax_lse = softmax_lse.transpose(2, 1).reshape(-1, self.num_heads)
        else:
            B = q.shape[0]
            output = torch.empty(B,
                                 self.num_heads,
                                 self.kv_lora_rank,
                                 dtype=q_dtype,
                                 device=q.device)
            softmax_lse = None
            gops.paged_attention_v1(out=output,
                                    query=q,
                                    key_cache=kv_c_and_k_pe_cache,
                                    value_cache=None,
                                    num_kv_heads=1,
                                    scale=self.scale,
                                    block_tables=decode_meta.block_table,
                                    seq_lens=decode_meta.seq_lens,
                                    block_size=kv_c_and_k_pe_cache.size(1),
                                    max_seq_len=decode_meta.max_decode_seq_len,
                                    alibi_slopes=None,
                                    kv_cache_dtype=self.kv_cache_dtype,
                                    k_scale_float=layer._k_scale_float,
                                    v_scale_float=layer._k_scale_float,
                                    out_scales=None,
                                    query_scales=q_scale)

        return output, softmax_lse
