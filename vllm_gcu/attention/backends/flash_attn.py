# SPDX-License-Identifier: Apache-2.0
"""
Attention layer with xFormers and PagedAttention.
refer to vllm.attention.backends.xformers
"""
from dataclasses import dataclass
from itertools import accumulate
from typing import Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.backends.utils import (
    PAD_SLOT_ID,
    CommonAttentionState,
    CommonMetadataBuilder,
    get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args,
    is_block_tables_empty,
    compute_slot_mapping_start_idx,
    compute_slot_mapping,
)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.attention.backends.xformers import XFormersImpl, XFormersMetadata
from vllm.logger import init_logger
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from vllm_gcu.attention.ops.paged_attn import PagedAttention

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

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

        #change here
        if self.max_decode_query_len > 1:
            query_start_loc = (
                None
                if self.query_start_loc is None
                else self.query_start_loc[self.num_prefills:] -
                                self.query_start_loc[self.num_prefills]
            )
            # seq_start_loc = (
            #     None
            #     if self.seq_start_loc is None
            #     else self.seq_start_loc[self.num_prefills:]
            # )
        else:
            query_start_loc=None
            # seq_start_loc=None
            
        # print("check")
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = GCUXFormersMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=query_start_loc,
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

        torch.ops._C.advance_step_flashattn(
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

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                # change here
                # assert query_len == 1, (
                #     "seq_len: {}, context_len: {}, query_len: {}".format(
                #         seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        # change here
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        return self._metadata_cls(  # type: ignore
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=True,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )

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

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
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
                assert out.shape == output[:num_prefill_query_tokens].shape
                output[:num_prefill_query_tokens] = out
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
                    # prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window,
                    layer._k_scale,
                    layer._v_scale,
                )
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            assert (
                attn_type != AttentionType.ENCODER_ONLY
            ), "Encoder-only models should not have decode metadata."
            if decode_meta.max_decode_query_len > 1:
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support max_decode_query_len > 1"
                )
                from flash_attn.vllm_flash_attn import flash_attn_varlen_func
                
                num_blocks, num_kv_heads, head_size, block_size = value_cache.shape[0],value_cache.shape[1], value_cache.shape[2], value_cache.shape[3]

                # # gcu:原始存，是按照[num_blocks, num_kv_heads, block_size, head_size]的顺序存的。
                key_cache = torch.as_strided(key_cache, size=(num_blocks, block_size, num_kv_heads, head_size), stride=(head_size*block_size*num_kv_heads,  head_size, head_size*block_size, 1))
                value_cache = torch.as_strided(value_cache, size=(num_blocks, block_size, num_kv_heads, head_size), stride=(head_size*block_size*num_kv_heads,  head_size, head_size*block_size, 1))

                flash_attn_varlen_func(
                    q=decode_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_decode_query_len,
                    seqused_k=decode_meta.seq_lens_tensor,
                    max_seqlen_k=decode_meta.max_decode_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=0.0,
                    block_table=decode_meta.block_tables,
                    out=output[num_prefill_query_tokens:],
                    fa_version=2,
                )

            else:
                (
                    seq_lens_arg,
                    max_seq_len_arg,
                    block_tables_arg,
                ) = get_seq_len_block_table_args(decode_meta, False, attn_type)

                output[num_prefill_query_tokens:] = PagedAttention.forward_decode(
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

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)
