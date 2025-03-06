#!/usr/bin/env python
# coding=utf-8
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Dict, List, Optional, T, Tuple, Type

import torch
import vllm.envs as envs

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.attention.backends.mla.utils import MLACommonImpl, MLACommonMetadata
from vllm.attention.backends.triton_mla import TritonMLAState
from vllm.attention.backends.utils import (
    compute_slot_mapping,
    compute_slot_mapping_start_idx,
    is_block_tables_empty,
    PAD_SLOT_ID,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    current_platform_fp8_dtype,
    is_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_quantize
from vllm.model_executor.model_loader.loader import device_loading_context

from vllm.multimodal import MultiModalPlaceholderMap
from vllm.utils import async_tensor_h2d, make_tensor_with_pad

import vllm_gcu.kernels._custom_ops as ops


class GCUMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GCU-MLA"

    @staticmethod
    def get_impl_cls() -> Type["GCUMLAImpl"]:
        return GCUMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return GCUMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["GCUMLAMetadataBuilder"]:
        return GCUMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["GCUMLAState"]:
        return GCUMLAState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        ops.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        # ops.copy_blocks_mla(kv_caches, src_to_dists)
        raise NotImplementedError

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [576]


class GCUMLAState(TritonMLAState):
    def graph_capture_get_metadata_for_batch(
        self, batch_size: int, is_encoder_decoder_model: bool = False
    ):
        if is_encoder_decoder_model:
            raise NotImplementedError(
                "GCUMLAState does not support encoder/decoder yet"
            )

        return super().graph_capture_get_metadata_for_batch(
            batch_size, is_encoder_decoder_model
        )

    def get_graph_input_buffers(
        self, attn_metadata, is_encoder_decoder_model: bool = False
    ):
        if is_encoder_decoder_model:
            raise NotImplementedError(
                "GCUMLAState does not support encoder/decoder yet"
            )

        return super().get_graph_input_buffers(attn_metadata, is_encoder_decoder_model)

    def prepare_graph_input_buffers(
        self, input_buffers, attn_metadata, is_encoder_decoder_model: bool = False
    ):
        if is_encoder_decoder_model:
            raise NotImplementedError(
                "GCUMLAState does not support encoder/decoder yet"
            )

        return super().prepare_graph_input_buffers(
            input_buffers, attn_metadata, is_encoder_decoder_model
        )


@dataclass
class GCUMLAMetadata(MLACommonMetadata):
    seq_lens: Optional[List[int]]
    seq_lens_tensor: Optional[torch.Tensor]
    max_prefill_seq_len: int
    max_decode_seq_len: int
    context_lens_tensor: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    use_cuda_graph: bool
    max_query_len: Optional[int] = None
    max_decode_query_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None
    seq_start_loc: Optional[torch.Tensor] = None
    _cached_prefill_metadata: Optional["GCUMLAMetadata"] = None
    _cached_decode_metadata: Optional["GCUMLAMetadata"] = None
    num_prefill_tokens: int
    num_kv_splits: int = 4
    attn_logits: Optional[torch.Tensor] = None
    req_idx: Optional[torch.Tensor] = None
    head_dim: Optional[int] = None

    def __post_init__(self):
        supported_head_sizes = GCUMLABackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f"received {self.head_dim}.",
            )

    @property
    def prefill_metadata(self) -> Optional["GCUMLAMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None

        # Compute some attn_metadata fields which default to None
        query_start_loc = (
            None
            if self.query_start_loc is None
            else self.query_start_loc[: self.num_prefills + 1]
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
        seq_start_loc = (
            None
            if self.seq_start_loc is None
            else self.seq_start_loc[: self.num_prefills + 1]
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
        input_positions = (
            None
            if self.input_positions is None
            else self.input_positions[: self.num_prefill_tokens]
        )

        self._cached_prefill_metadata = GCUMLAMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            input_positions=input_positions,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            head_dim=self.head_dim,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["GCUMLAMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.seq_lens_tensor is not None

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
        input_positions = (
            None
            if self.input_positions is None
            else self.input_positions[self.num_prefill_tokens :]
        )

        self._cached_decode_metadata = GCUMLAMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            # Batch may be composed of prefill|decodes, adjust query start
            # indices to refer to the start of decodes. E.g.
            # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
            query_start_loc=(
                (
                    self.query_start_loc[self.num_prefills :]
                    - self.query_start_loc[self.num_prefills]
                )
                if self.query_start_loc is not None
                else None
            ),
            seq_start_loc=(
                self.seq_start_loc[self.num_prefills :]
                if self.seq_start_loc is not None
                else None
            ),
            context_lens_tensor=None,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            input_positions=input_positions,
            head_dim=self.head_dim,
        )
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
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
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

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1,)
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1,)

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries,)

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

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
            input_positions=self.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
        )


class GCUMLAMetadataBuilder(AttentionMetadataBuilder[GCUMLAMetadata]):
    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.input_positions: List[int] = []
        self.multimodal_placeholder_maps: Dict[str, MultiModalPlaceholderMap] = (
            defaultdict(MultiModalPlaceholderMap)
        )
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

    def _add_seq_group(
        self,
        inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
        chunked_prefill_enabled: bool,
        prefix_cache_hit: bool,
    ):
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (
            seq_id,
            token_len,
            seq_len,
            curr_seq_len,
            query_len,
            context_len,
            curr_sliding_window_block,
            input_positions,
        ) in zip(
            inter_data.seq_ids,
            [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens,
            inter_data.seq_lens,
            inter_data.query_lens,
            inter_data.context_lens,
            inter_data.curr_sliding_window_blocks,
            inter_data.input_positions,
        ):
            self.input_positions.extend(input_positions)
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif (
                chunked_prefill_enabled or not is_prompt
            ) and block_tables is not None:
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window
            )
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )

    def _get_graph_runner_block_tables(
        self, num_seqs: int, block_tables: List[List[int]]
    ) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True
        )

    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        cuda_graph_pad_size: int,
        batch_size: int,
    ):
        """Build attention metadata with on-device tensors.

        Args:                                                                                                                                             seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any(
            [
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ]
        )
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(
                inter_data, self.input_builder.chunked_prefill_enabled, prefix_cache_hit
            )

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills :]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables
            )
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        context_lens_tensor = async_tensor_h2d(
            self.context_lens, torch.int, device, self.runner.pin_memory
        )
        seq_lens_tensor = async_tensor_h2d(
            seq_lens, torch.int, device, self.runner.pin_memory
        )
        input_positions = async_tensor_h2d(
            self.input_positions, torch.long, device, self.runner.pin_memory
        )
        slot_mapping_tensor = async_tensor_h2d(
            self.slot_mapping, torch.long, device, self.runner.pin_memory
        )
        query_start_loc_tensor = async_tensor_h2d(
            query_start_loc, torch.int32, device, self.runner.pin_memory
        )
        seq_start_loc_tensor = async_tensor_h2d(
            seq_start_loc, torch.int32, device, self.runner.pin_memory
        )
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in self.multimodal_placeholder_maps.items()
        }

        return GCUMLAMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=True,
            input_positions=input_positions,
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
            num_kv_splits=4,  # TODO(lucas) add heuristic
            head_dim=self.runner.model_config.get_head_size(),
        )


class GCUMLAImpl(MLACommonImpl[GCUMLAMetadata]):

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
                "GCUMLAImpl does not support one of the following: alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention are not implemented for GCUMLAImpl"
            )

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError("output is not yet supported for MLAImplBase")

        is_decode = attn_metadata.decode_metadata is not None
        is_prefill = attn_metadata.prefill_metadata is not None

        if is_decode and is_prefill:
            raise NotImplementedError(
                "chunked prefill is not supported for MLAImplBase"
            )

        # Restore head dim (for rotary embedding)
        k_pe = k_pe.unsqueeze(1)
        assert hasattr(attn_metadata, "input_positions")
        rope_fn = self.rotary_emb if self.use_yarn_rope else self.apply_pure_rope

        if is_decode:
            q_nope = self._q_proj_and_k_up_proj(hidden_states_or_q_c)
            q_pe = torch.matmul(hidden_states_or_q_c, self.W_QR).view(
                -1, self.num_heads, self.qk_rope_head_dim
            )
            q_pe, k_pe = rope_fn(attn_metadata.input_positions, q_pe, k_pe)
        else:
            assert is_prefill
            q = self.q_proj(hidden_states_or_q_c)[0].view(
                -1, self.num_heads, self.qk_head_dim
            )

            # TODO(lucas): there must be a nicer way to write this line
            q[..., self.qk_nope_head_dim :], k_pe = rope_fn(
                attn_metadata.input_positions, q[..., self.qk_nope_head_dim :], k_pe
            )

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

        if attn_metadata.prefill_metadata is not None:
            return self._forward_prefill(q, k_c_normed, k_pe, attn_metadata)

        if attn_metadata.decode_metadata is not None:
            return self._forward_decode(q_nope, q_pe, kv_cache, attn_metadata)

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_scale_group_shapes_for_fp8(
            layer: LinearBase,
        ) -> Tuple[Tuple[int, int], Tuple[int, int]]:

            if isinstance(layer.quant_method, Fp8LinearMethod):
                if layer.quant_method.block_quant:
                    weight_block_size = (
                        layer.quant_method.quant_config.weight_block_size
                    )
                    # per-token-group (1, X), block-quantized (X, Y)
                    return (1, weight_block_size[-1]), weight_block_size
                else:
                    return (-1, -1), (-1, -1)  # per-tensor, per-tensor

            elif isinstance(
                layer.quant_method, CompressedTensorsLinearMethod
            ) and isinstance(layer.scheme, CompressedTensorsW8A8Fp8):
                # this is hacky but we always assume the for
                # CompressedTensorsW8A8Fp8 the input is dynamic per-token
                # we ignore if it is static-per-tensor since we are going to
                # requantize after later anyways
                from compressed_tensors.quantization import QuantizationStrategy

                strategy = layer.scheme.strategy
                if strategy == QuantizationStrategy.TENSOR:
                    return (1, -1), (-1, -1)  # per-token, per-tensor
                elif strategy == QuantizationStrategy.CHANNEL:
                    return (1, -1), (-1, 1)  # per-token, per-channel
                else:
                    raise NotImplementedError(
                        f"QuantizationStrategy.{strategy} is not supported for "
                        "fp8 MLA, please run with VLLM_MLA_DISABLE=1"
                    )

            else:
                raise NotImplementedError(
                    "Can't determine scale group shapes for "
                    f"{layer.quant_method}, please run with VLLM_MLA_DISABLE=1"
                )

        def get_layer_weight(layer):
            if hasattr(layer, "weight"):
                return layer.weight
            elif hasattr(layer, "qweight"):
                return layer.qweight
            else:
                raise AttributeError(f"Layer '{layer}' has neither weight nor qweight")

        def process_attention_linear(layer: LinearBase):
            if quant_method := getattr(layer, "quant_method", None):
                if isinstance(quant_method, QuantizeMethodBase):
                    with device_loading_context(layer, get_layer_weight(layer).device):
                        quant_method.process_weights_after_loading(layer)

        def get_and_maybe_dequant_weights(layer: LinearBase):
            process_attention_linear(layer)
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                eye = torch.eye(
                    (
                        layer.input_size_per_partition
                        if hasattr(layer, "input_size_per_partition")
                        else layer.input_size
                    ),
                    dtype=act_dtype,
                    device=get_layer_weight(layer).device,
                )
                dequant_weights = layer.quant_method.apply(layer, eye, bias=None)
                del eye
                return dequant_weights.T
            return layer.weight

        weight_dtype = get_layer_weight(self.kv_b_proj).dtype
        assert get_layer_weight(self.o_proj).dtype == weight_dtype
        assert get_layer_weight(self.q_proj).dtype == weight_dtype

        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        q_proj_weight = get_and_maybe_dequant_weights(self.q_proj).T.view(
            -1, self.num_heads, self.qk_head_dim
        )

        # can be W_Q or W_UQ depending q_lora_rank, the former if
        # q_lora_rank is None, the latter otherwise. From the Attention backend
        # perspective though we call these both W_Q and rely on the layer
        # to pass in the correct matrix
        W_Q = q_proj_weight[..., : self.qk_nope_head_dim]
        self.W_QR = (
            q_proj_weight[..., self.qk_nope_head_dim :]
            .flatten(start_dim=1)
            .contiguous()
        )

        # W_QR is small so for simplicity we dont bother requantizing it
        self.W_QR = self.W_QR.to(act_dtype)

        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            requantization_enabled = not envs.VLLM_MLA_DISABLE_REQUANTIZATION
            if is_fp8(weight_dtype) and requantization_enabled:
                # This assumes it wise to requantize using the same group shapes
                # (i.e. strategy, per-tensor, per-channel, block etc.) that the
                # weights were originally quantized
                requant_input_group_shape, requant_weight_group_shape = (
                    get_scale_group_shapes_for_fp8(self.q_proj)
                )
                assert (
                    requant_input_group_shape,
                    requant_weight_group_shape,
                ) == get_scale_group_shapes_for_fp8(self.kv_b_proj)
                assert (
                    requant_input_group_shape,
                    requant_weight_group_shape,
                ) == get_scale_group_shapes_for_fp8(self.o_proj)
                self.reqaunt_input_group_shape = requant_input_group_shape
                self.reqaunt_weight_group_shape = requant_weight_group_shape

            #
            # Perform matrix-absorption following
            #     https://github.com/flashinfer-ai/flashinfer/pull/551
            # for decode, as a result we end up with absorbed weights for decode
            # and another copy of raw weights for prefill.
            #
            self.W_UK, self.W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            # We absorb `W_UK` into `W_Q` resulting in either W_Q_UK or W_UQ_UK
            # depending q_lora_rank, the former if q_lora_rank is None, the
            # latter otherwise
            # basically if q_lora_rank is none we are absorbing into q_proj
            # instead of UQ
            W_Q_UK = (
                torch.einsum("qnd,lnd -> qnl", W_Q, W_UK)
                .flatten(start_dim=1)
                .contiguous()
            )

            if is_fp8(weight_dtype) and requantization_enabled:
                W_Q_UK, W_Q_UK_scales = scaled_quantize(
                    W_Q_UK,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype,
                )
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_Q_UK = W_Q_UK.T.contiguous()
                self.W_Q_UK_scales = W_Q_UK_scales.T.contiguous()
            else:
                self.W_Q_UK = W_Q_UK.to(act_dtype)

            W_O = get_and_maybe_dequant_weights(self.o_proj).view(
                -1, self.num_heads, self.v_head_dim
            )
            W_UV_O = (
                torch.einsum("lnd,hnd -> nlh", W_UV, W_O)
                .flatten(start_dim=0, end_dim=1)
                .contiguous()
            )

            if is_fp8(weight_dtype) and requantization_enabled:
                W_UV_O, W_UV_O_scales = scaled_quantize(
                    W_UV_O,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype,
                )
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_UV_O = W_UV_O.T.contiguous()
                self.W_UV_O_scales = W_UV_O_scales.T.contiguous()
            else:
                self.W_UV_O = W_UV_O.to(act_dtype)

            self.tp_size = get_tensor_model_parallel_world_size()
        else:
            if is_fp8(weight_dtype):
                raise NotImplementedError("Currently fp8 requires matrix absorption")

            self.W_UV = W_UV
            self.W_UK = W_UK
            self.W_Q = W_Q.flatten(start_dim=1)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: GCUMLAMetadata,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, GCUMLAMetadata)

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
        attn_metadata: GCUMLAMetadata,
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

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                self.num_heads,
                attn_metadata.num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
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
