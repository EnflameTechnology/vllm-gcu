#!/usr/bin/env python
# coding=utf-8
import itertools
from itertools import accumulate
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

import torch
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)

from vllm.platforms import current_platform
from vllm.utils import async_tensor_h2d, cdiv, make_tensor_with_pad, round_down
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.attention.backends.abstract import AttentionMetadata


from vllm import envs
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
import vllm_gcu.kernels._custom_ops as ops
if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class GCUMLABackend(MLACommonBackend):

    accept_output_buffer: bool = True

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
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return GCUMLACommonMetadata

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


@dataclass
class GCUMLACommonMetadata(MLACommonMetadata):
    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        super().advance_step(model_input, sampled_token_ids, block_size,
                             num_seqs, num_queries, turn_prefills_into_decodes)
        if not self.input_positions is model_input.input_positions:
            # NOTE: input positions in model_input and attn_metadata
            # are different obj in driver worker, same obj in other workers.
            self.input_positions.add_(1)

class GCUMLACommonMetadataBuilder(MLACommonMetadataBuilder):
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
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])

        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
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

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        input_positions = async_tensor_h2d(self.input_positions, torch.long,
                                           device, self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)

        context_chunk_cu_seq_lens = None
        context_chunk_starts = None
        context_chunk_seq_tot = None
        context_chunk_max_seq_lens = None

        if (self.chunked_prefill_enabled or self.enable_prefix_caching) \
                and self.num_prefills > 0 \
                and context_lens_tensor is not None \
                and context_lens_tensor[:self.num_prefills].max() > 0:

            # NOTE: it is recommend you read the `Chunked Prefill` section in
            # the comment at the top of the file before trying to understand
            # the following code

            num_prefills_with_context = \
                (context_lens_tensor[:self.num_prefills] > 0).sum().item()

            # currently we allocate an equal amount of workspace for each
            # prefill in the batch, we could probably use a more advanced
            # algorithm here and allocate more workspace to prefills with
            # longer context lengths
            max_context_chunk = \
                self.context_chunk_workspace_size // num_prefills_with_context

            # align max_context_chunk to page_size by rounding down,
            # currently the `gather_cache` kernel cannot handle
            # `context_chunk_starts` that are not aligned to page_size
            max_context_chunk = round_down(max_context_chunk, self.page_size)
            assert max_context_chunk > 0
            num_chunks = cdiv(context_lens_tensor.max(), max_context_chunk)

            # if `max_context_chunk = 256`, `num_chunks = 3`, and
            #   `num_prefills_with_context = 4`, create a tensor that looks like
            #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
            context_chunk_starts = \
                torch.arange(num_chunks, device=device, dtype=torch.int32)\
                .unsqueeze(1).expand(-1, self.num_prefills)\
                * max_context_chunk
            chunk_ends = torch.min(context_lens_tensor[:self.num_prefills]
                                   .unsqueeze(0), context_chunk_starts + max_context_chunk)
            chunk_seq_lens = (chunk_ends - context_chunk_starts).clamp(min=0)
            _context_chunk_cu_seq_lens = chunk_seq_lens.cumsum(dim=1).to(
                torch.int32)
            zero = torch.zeros(num_chunks, dtype=torch.int32, device=device)\
                .unsqueeze(-1)
            context_chunk_cu_seq_lens = \
                torch.cat([zero, _context_chunk_cu_seq_lens], dim=1)
            context_chunk_max_seq_lens = \
                chunk_seq_lens.max(dim=1).values.tolist()
            context_chunk_seq_tot = chunk_seq_lens.sum(dim=1).tolist()
            assert max(context_chunk_seq_tot) <= \
                self.context_chunk_workspace_size

        return self.runner.attn_backend.make_metadata(
            # Required by ModelRunner
            use_cuda_graph=use_captured_graph,  # Not Attention Related
            # Required by Attention Metadata
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            # Required by Attention Metadata (not used)
            multi_modal_placeholder_index_maps=None,  # Not Attention Related
            enable_kv_scales_calculation=False,
            # MLACommonMetadata
            input_positions=input_positions,
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
            head_dim=self.runner.model_config.get_head_size(),
            is_profile_run=self.runner.in_profile_run,
            # MLACommonMetadata Chunk prefill specific
            context_chunk_cu_seq_lens=context_chunk_cu_seq_lens,
            context_chunk_starts=context_chunk_starts,
            context_chunk_seq_tot=context_chunk_seq_tot,
            context_chunk_max_seq_lens=context_chunk_max_seq_lens,
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

        self._pad_v = False # only for flash attn

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
            if output is not None:
                return output
            else:
                return torch.empty(
                    [0, self.o_proj.output_size],
                    dtype=hidden_states_or_q_c.dtype,
                    device=hidden_states_or_q_c.device,
                )

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
        # === VLLM_GCU MODIFY START ===
        if current_platform.get_device_capability()[0] == 13:
            return self._forward_prefill_xformers(
                q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata
            )
        from flash_attn.vllm_flash_attn import flash_attn_varlen_func
        self.flash_attn_varlen_func = flash_attn_varlen_func
        is_hip = False
        is_vllm_fa = False
        # === VLLM_GCU MODIFY END ===

        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None

        has_context = prefill_metadata.context_lens_tensor is not None \
            and prefill_metadata.context_lens_tensor.max() > 0

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        # === VLLM_GCU MODIFY START ===
        if self._pad_v:
            v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                               value=0)
        else:
            v_padded = v
        # === VLLM_GCU MODIFY END ===

        if is_hip and envs.VLLM_USE_TRITON_FLASH_ATTN and not has_context:
            output = self.triton_fa_func(
                q,
                k,
                v_padded,
                None,
                prefill_metadata.query_start_loc,
                prefill_metadata.query_start_loc,
                prefill_metadata.max_prefill_seq_len,
                prefill_metadata.max_prefill_seq_len,
                True,  # causal
                self.scale,
                None,  # attn_mask is None unless applying ALiBi mask
            )
            ## triton flash attention always return 2 objects
            if not has_context:
                output = output[0]
        elif is_vllm_fa:
            output = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v_padded,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.query_start_loc,
                max_seqlen_q=prefill_metadata.max_prefill_seq_len,
                max_seqlen_k=prefill_metadata.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                return_softmax_lse=has_context,
            )
        else:
            output = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v_padded,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.query_start_loc,
                max_seqlen_q=prefill_metadata.max_prefill_seq_len,
                max_seqlen_k=prefill_metadata.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                return_attn_probs=has_context,
            )

        if has_context:
            # ROCm flash_attn_varlen_func will return 3 objects instead of 2
            suffix_output, suffix_lse, *rest = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # slice by `:v.shape[-1]` in order to remove v headdim padding
        # === VLLM_GCU MODIFY START ===
        if self._pad_v:
            output = output\
                .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])
        else:
            output = output.view(-1, self.num_heads * v.shape[-1])
        # === VLLM_GCU MODIFY END ===
        return self.o_proj(output)[0]

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ):
        from flash_attn.vllm_flash_attn import flash_attn_varlen_func
        self.flash_attn_varlen_func = flash_attn_varlen_func
        is_vllm_fa = False
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
        workspace = attn_metadata.context_chunk_workspace

        for i in range(iters):
            toks = prefill_metadata.context_chunk_seq_tot[i]

            ops.gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_tables,
                cu_seq_lens=prefill_metadata.context_chunk_cu_seq_lens[i],
                batch_size=prefill_metadata.num_prefills,
                seq_starts=prefill_metadata.context_chunk_starts[i],
            )

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

            # For MLA the v head dim is smaller than qk head dim so we pad
            # out v with 0s to match the qk head dim
            # === VLLM_GCU MODIFY START ===
            if self._pad_v:
                v_padded = torch.nn.functional.pad(v,
                                                   [0, q.shape[-1] - v.shape[-1]],
                                                   value=0)
            else:
                v_padded = v
            # === VLLM_GCU MODIFY END ===

            if is_vllm_fa:
                attn_output, attn_softmax_lse = self.flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v_padded,
                    cu_seqlens_q=prefill_metadata.query_start_loc,
                    cu_seqlens_k=prefill_metadata.context_chunk_cu_seq_lens[i],
                    max_seqlen_q=prefill_metadata.max_query_len,
                    max_seqlen_k=prefill_metadata.
                    context_chunk_max_seq_lens[i],
                    softmax_scale=self.scale,
                    causal=False,  # Context is unmasked
                    return_softmax_lse=True,
                )
            else:
                attn_output, attn_softmax_lse, _ = self.flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v_padded,
                    cu_seqlens_q=prefill_metadata.query_start_loc,
                    cu_seqlens_k=prefill_metadata.context_chunk_cu_seq_lens[i],
                    max_seqlen_q=prefill_metadata.max_query_len,
                    max_seqlen_k=prefill_metadata.
                    context_chunk_max_seq_lens[i],
                    softmax_scale=self.scale,
                    causal=False,  # Context is unmasked
                    return_attn_probs=True,
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
