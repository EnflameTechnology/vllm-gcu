# SPDX-License-Identifier: Apache-2.0
"""
Attention layer with xFormers and PagedAttention.
refer to vllm.attention.backends.xformers
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.backends.utils import (
    CommonAttentionState,
    CommonMetadataBuilder,
    get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args,
)
from vllm.attention.backends.xformers import XFormersImpl, XFormersMetadata
from vllm.logger import init_logger
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from vllm_gcu.attention.ops.paged_attn import PagedAttention

logger = init_logger(__name__)


class GCUXFormersBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "XFORMERS"

    @staticmethod
    def get_impl_cls() -> Type["GCUXFormersImpl"]:
        return GCUXFormersImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return GCUXFormersMetadata

    @staticmethod
    def get_builder_cls() -> Type["GCUXFormersMetadataBuilder"]:
        return GCUXFormersMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class GCUXFormersMetadata(XFormersMetadata):
    @property
    def prefill_metadata(self) -> Optional["GCUXFormersMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert (self.seq_lens is not None) or (self.encoder_seq_lens is not None)
        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        # Compute some attn_metadata fields which default to None
        query_start_loc = (
            None
            if self.query_start_loc is None
            else self.query_start_loc[: self.num_prefills + 1]
        )
        seq_start_loc = (
            None
            if self.seq_start_loc is None
            else self.seq_start_loc[: self.num_prefills + 1]
        )
        slot_mapping = (
            None
            if self.slot_mapping is None
            else self.slot_mapping[: self.num_prefill_tokens]
        )
        seq_lens = None if self.seq_lens is None else self.seq_lens[: self.num_prefills]
        seq_lens_tensor = (
            None
            if self.seq_lens_tensor is None
            else self.seq_lens_tensor[: self.num_prefills]
        )
        context_lens_tensor = (
            None
            if self.context_lens_tensor is None
            else self.context_lens_tensor[: self.num_prefills]
        )
        block_tables = (
            None
            if self.block_tables is None
            else self.block_tables[: self.num_prefills]
        )

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = GCUXFormersMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["GCUXFormersMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        # Compute some attn_metadata fields which default to None
        slot_mapping = (
            None
            if self.slot_mapping is None
            else self.slot_mapping[self.num_prefill_tokens :]
        )
        seq_lens_tensor = (
            None
            if self.seq_lens_tensor is None
            else self.seq_lens_tensor[self.num_prefills :]
        )
        block_tables = (
            None
            if self.block_tables is None
            else self.block_tables[self.num_prefills :]
        )

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = GCUXFormersMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
        )

        # Batch may be composed of prefill|decodes, adjust query start indices
        # to refer to the start of decodes when the two are split apart.
        # E.g. in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata

    def advance_step(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        sampled_token_ids: Optional[torch.Tensor],
        block_size: int,
        num_seqs: int,
        num_queries: int,
        turn_prefills_into_decodes: bool = False,
    ):
        from vllm_gcu.kernels import _custom_ops as ops

        # When using graph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
            # When Multi-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs,)

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs,)
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # only support decoder-only
        assert self.encoder_seq_lens is None
        assert self.encoder_seq_lens_tensor is None
        assert self.num_encoder_tokens is None or self.num_encoder_tokens == 0
        assert self.cross_slot_mapping is None
        assert self.cross_block_tables is None

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=sampled_token_ids,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
        )


class GCUXFormersMetadataBuilder(CommonMetadataBuilder[GCUXFormersMetadata]):

    _metadata_cls = GCUXFormersMetadata


class GCUXFormersImpl(XFormersImpl):
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: "GCUXFormersMetadata",
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.numel() == 0:
            return query.view(-1, self.num_heads * self.head_size)

        attn_type = self.attn_type
        if attn_type == AttentionType.ENCODER and (
            not attn_metadata.is_all_encoder_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder attention requires setting " "encoder metadata attributes."
            )

        elif attn_type == AttentionType.ENCODER_DECODER and (
            not attn_metadata.is_all_cross_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder/decoder cross-attention "
                "requires setting cross-attention "
                "metadata attributes."
            )

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        k_zero_float = getattr(layer, "_k_zero_float", 0.0)
        v_zero_float = getattr(layer, "_v_zero_float", 0.0)

        if attn_type != AttentionType.ENCODER and kv_cache.numel() > 0:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )

            if (key is not None) and (value is not None):

                if attn_type == AttentionType.ENCODER_DECODER:
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    updated_slot_mapping = attn_metadata.slot_mapping

                PagedAttention.write_to_paged_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    updated_slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale_float,
                    layer._v_scale_float,
                    k_zero_float,
                    v_zero_float,
                )
        (num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens) = (
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        )
        if num_prefill_query_tokens and num_decode_query_tokens:
            output = torch.empty_like(query)
        decode_query = query[num_prefill_query_tokens:]
        query = query[:num_prefill_query_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_kv_tokens]
            value = value[:num_prefill_kv_tokens]

        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                out = self._run_memory_efficient_xformers_forward(
                    query, key, value, prefill_meta, attn_type=attn_type
                )
            else:
                assert (
                    attn_type != AttentionType.ENCODER_ONLY
                ), "Encoder-only models should not have prefix attention."

                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.max_query_len is not None

                out = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window,
                    layer._k_scale,
                    layer._v_scale,
                )

            if num_decode_query_tokens:
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out
            else:
                output = out

        if decode_meta := attn_metadata.decode_metadata:
            assert (
                attn_type != AttentionType.ENCODER_ONLY
            ), "Encoder-only models should not have decode metadata."

            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = get_seq_len_block_table_args(decode_meta, False, attn_type)

            decode_output = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                layer._k_scale_float,
                layer._v_scale_float,
                k_zero_float=k_zero_float,
                v_zero_float=v_zero_float,
                out_scales=layer.out_scales if hasattr(layer, "out_scales") else None,
            )

            if num_prefill_query_tokens:
                output[num_prefill_query_tokens:] = decode_output
            else:
                output = decode_output

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)
