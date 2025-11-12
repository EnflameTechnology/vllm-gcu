#!/usr/bin/env python
# coding=utf-8
from typing import Optional
import copy
import torch

from vllm.config import VllmConfig, CUDAGraphMode
from vllm.utils import cdiv
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.v1.spec_decode.eagle import EagleProposer, PADDING_SLOT_ID
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm_gcu.utils import set_gcu_forward_context

import triton_gcu.triton
import triton
import triton.language as tl

import numpy as np


@triton.jit
def prepare_next_token_ids_kernel(
    sampled_ptr,  # *int32,  [num_reqs, max_gen_len]
    backup_ptr,  # *int32,  [num_reqs]
    discard_request_indices_ptr, # *int32 [num_reqs]
    vocab_size,  # int32
    max_gen_len,  # int32
    num_discarded_requests, # int32
    num_reqs,  # int32
    next_ptr,  # *int32, [num_reqs]
    count_ptr,  # *int32, [num_reqs]
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    discard = 0
    for i in tl.range(num_discarded_requests):
        discard_request_indices = tl.load(discard_request_indices_ptr + i)
        discard = 1 if discard_request_indices== pid else 0

    row_start = pid * max_gen_len

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < max_gen_len
    tokens = tl.load(sampled_ptr + row_start + cols, mask=mask, other=-1)
    tokens = tl.where(discard == 1, -1, tokens)

    valid = (tokens != -1) & (tokens < vocab_size)
    cnt = tl.sum(valid.to(tl.int32))
    tl.store(count_ptr + pid, cnt)

    idxs = tl.where(valid, cols, -1)
    rightmost_idx = tl.max(idxs, axis=0)
    has_valid = rightmost_idx != -1

    chosen = tl.load(sampled_ptr + row_start +
                     rightmost_idx) if has_valid else -1
    backup = tl.load(backup_ptr + pid)
    final = tl.where(has_valid, chosen, backup)
    tl.store(next_ptr + pid, final)

@triton.jit
def prepare_inputs_padded_kernel(
    cu_num_draft,       # [num_seqs]
    valid_sampled,      # [num_seqs]
    query_start_loc,    # [num_seqs+1]
    out_tok_idx,        # [num_seqs]
    num_seqs,
    arange_ptr,
    token_indices_ptr,
    total_num_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid > 0:
        return
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_num_tokens
    arange = tl.load(arange_ptr+offs, mask=mask)
    tl.store(token_indices_ptr+offs, arange, mask=mask)

    mask = (offs < num_seqs) & (offs > 0)
    d1 = tl.load(cu_num_draft + offs, mask=mask)
    d0 = tl.load(cu_num_draft + offs - 1, mask=mask)
    mask = offs == 0
    d2 = tl.load(cu_num_draft + offs, mask=mask)
    d = d2 + d1 - d0

    mask2 = offs < num_seqs
    vs = tl.load(valid_sampled + offs, mask=mask2)
    rej = tl.where(d > 0, d + 1 - vs, 0)

    q1 = tl.load(query_start_loc + offs + 1, mask=mask2)
    tok_idx = q1 - 1 - rej
    tl.store(out_tok_idx + offs, tok_idx, mask=mask2)


class EagleProposerWithGraph(EagleProposer):

    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 runner=None,
                 prepare_next_token_ids_padded_event=None):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self.prepare_next_token_ids_padded_event = prepare_next_token_ids_padded_event

        self.cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        self.runtime_mode = self.cudagraph_mode.decode_mode()
        self.cudagraph_dispatcher1 = CudagraphDispatcher(self.vllm_config)

        mtp_config = copy.deepcopy(vllm_config)
        mtp_config.scheduler_config.cuda_graph_sizes = list(
            dict.fromkeys(
                cdiv(x, (1 + self.num_speculative_tokens))
                for x in mtp_config.compilation_config.cudagraph_capture_sizes
                if x > self.num_speculative_tokens))
        mtp_config.compilation_config.cudagraph_capture_sizes = None
        mtp_config._set_cudagraph_sizes()

        self.mtp_config = mtp_config
        self.cudagraph_dispatcher2 = CudagraphDispatcher(self.mtp_config)
        self.mtp_cudagraph_batch_sizes = list(
            reversed(
                self.mtp_config.compilation_config.cudagraph_capture_sizes))

        self.slot_mapping = torch.zeros(self.max_num_tokens,
                                        device=device,
                                        dtype=torch.int32)

    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        self.mtp_config.compilation_config.static_forward_context = (
            self.vllm_config.compilation_config.static_forward_context)

        if self.cudagraph_mode.has_full_cudagraphs():
            self.model1 = CUDAGraphWrapper(self.model,
                                           self.vllm_config,
                                           runtime_mode=CUDAGraphMode.FULL)
            self.model2 = CUDAGraphWrapper(self.model,
                                           self.mtp_config,
                                           runtime_mode=CUDAGraphMode.FULL)
        else:
            self.model1 = self.model2 = self.model

    def _get_positions(self, num_tokens: int):
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        self.positions[:num_tokens] = positions

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embeds: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        self.slot_mapping[:num_tokens].copy_(common_attn_metadata.slot_mapping)
        self.slot_mapping[num_tokens:].fill_(PADDING_SLOT_ID)
        common_attn_metadata.slot_mapping = self.slot_mapping[:num_tokens]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.model1, Eagle3LlamaForCausalLM) or (
                isinstance(self.model1, CUDAGraphWrapper)
                and isinstance(self.model1.unwrap(), Eagle3LlamaForCausalLM))
            target_hidden_states = self.model1.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0)
        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                ))
        else:
            draft_indexer_metadata = None
        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        num_input_tokens = num_tokens
        if self.runtime_mode != CUDAGraphMode.NONE and \
            num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)

        uniform_decode = num_tokens == batch_size * (
            self.num_speculative_tokens + 1)
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                           uniform_decode=uniform_decode)
        cudagraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher1.dispatch(
            batch_descriptor)

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.is_multimodal_model:
            mm_embeds, is_mm_embed = mm_embeds or (None, None)

            self.inputs_embeds[:num_tokens] = self.model1.get_input_embeddings(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        with set_gcu_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
        ):
            ret_hidden_states = self.model1(
                input_ids=input_ids,
                positions=self._get_positions(num_input_tokens),
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model1.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            return draft_token_ids.view(-1, 1)

        positions = target_positions[last_token_indices]
        if self.method in ("deepseek_mtp", "ernie_mtp", "longcat_flash_mtp"):
            hidden_states = self.hidden_states[last_token_indices]
        else:
            hidden_states = hidden_states[last_token_indices]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            raise ValueError("tree attention is not support")
            # # Draft using tree attention.
            # draft_token_ids_list = self.propose_tree(
            #     batch_size=batch_size,
            #     logits=logits,
            #     positions=positions,
            #     hidden_states=hidden_states,
            #     common_attn_metadata=common_attn_metadata,
            # )
            # # [batch_size, num_tree_tokens]
            # return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = logits.argmax(dim=-1)

        if self.allowed_attn_types is not None and not isinstance(
                attn_metadata, self.allowed_attn_types):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}")

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        input_batch_size = batch_size
        if self.runtime_mode != CUDAGraphMode.NONE and \
            batch_size <= self.mtp_cudagraph_batch_sizes[-1]:
            input_batch_size = self.mtp_config.pad_for_cudagraph(batch_size)

        batch_descriptor = BatchDescriptor(num_tokens=input_batch_size,
                                           uniform_decode=True)
        cudagraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher2.dispatch(
            batch_descriptor)

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[:batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[:batch_size + 1]).clone()
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            positions += 1
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            positions)

            # Increment the sequence lengths.
            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens_cpu += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.

            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len,
                                                       1)

            common_attn_metadata.num_computed_tokens_cpu = (
                common_attn_metadata.seq_lens_cpu - 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            common_attn_metadata.slot_mapping = self.slot_mapping[:batch_size]
            common_attn_metadata.slot_mapping.copy_(
                block_ids * self.block_size +
                clamped_positions % self.block_size)
            self.slot_mapping[batch_size:].fill_(PADDING_SLOT_ID)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID)

            # Rebuild attention metadata
            attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
                common_attn_metadata=common_attn_metadata,
                draft_index=token_index + 1)
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.is_multimodal_model:
                self.inputs_embeds[:
                                   batch_size] = self.model2.get_input_embeddings(
                                       input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            with set_gcu_forward_context(
                    per_layer_attn_metadata,
                    self.mtp_config,
                    num_tokens=input_batch_size,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
            ):
                ret_hidden_states = self.model2(
                    input_ids=input_ids,
                    positions=self._get_positions(input_batch_size),
                    hidden_states=self.hidden_states[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
            hidden_states = hidden_states[:batch_size]
            logits = self.model2.compute_logits(
                last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        common_attn_metadata: Optional[CommonAttentionMetadata] = None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if cudagraph_runtime_mode == CUDAGraphMode.FULL:
            assert common_attn_metadata is not None
            spec_common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=common_attn_metadata.query_start_loc,
                seq_lens=common_attn_metadata.seq_lens,
                query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
                seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
                num_computed_tokens_cpu=common_attn_metadata.
                num_computed_tokens_cpu,
                num_reqs=common_attn_metadata.num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                slot_mapping=self.slot_mapping[:num_tokens],
                causal=True,
            )
        else:
            spec_common_attn_metadata = None

        # mtp1
        per_layer_attn_metadata = None
        if spec_common_attn_metadata is not None:
            per_layer_attn_metadata = {}
            if self.attn_metadata_builder is None:
                attn_metadata_builder = self._get_attention_metadata_builder()
            else:
                attn_metadata_builder = self.attn_metadata_builder
            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=spec_common_attn_metadata, draft_index=0)
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            uniform_decode=cudagraph_runtime_mode == CUDAGraphMode.FULL)
        cudagraph_runtime_mode1, batch_descriptor = self.cudagraph_dispatcher1.dispatch(
            batch_descriptor)

        with set_gcu_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
        ):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            self.model1(
                input_ids=input_ids,
                positions=self.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )

        if self.num_speculative_tokens == 1:
            return

        batch_size = cdiv(num_tokens, (self.num_speculative_tokens + 1))

        # mtp>1
        per_layer_attn_metadata = None
        if spec_common_attn_metadata is not None:
            per_layer_attn_metadata = {}
            spec_common_attn_metadata.num_actual_tokens = batch_size
            spec_common_attn_metadata.max_query_len = 1
            spec_common_attn_metadata.query_start_loc = self.arange[:batch_size
                                                                    + 1]
            spec_common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[:batch_size + 1]).clone()
            spec_common_attn_metadata.slot_mapping = self.slot_mapping[:
                                                                       batch_size]
            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=spec_common_attn_metadata, draft_index=1)
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        batch_descriptor = BatchDescriptor(
            num_tokens=batch_size,
            uniform_decode=cudagraph_runtime_mode == CUDAGraphMode.FULL)
        cudagraph_runtime_mode2, batch_descriptor = self.cudagraph_dispatcher2.dispatch(
            batch_descriptor)

        for token_index in range(self.num_speculative_tokens - 1):
            with set_gcu_forward_context(
                    per_layer_attn_metadata,
                    self.mtp_config,
                    num_tokens=batch_size,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
            ):
                if self.is_multimodal_model:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:batch_size]
                else:
                    input_ids = self.input_ids[:batch_size]
                    inputs_embeds = None

                self.model2(
                    input_ids=input_ids,
                    positions=self.positions[:batch_size],
                    hidden_states=self.hidden_states[:batch_size],
                    inputs_embeds=inputs_embeds,
                )

    def prepare_next_token_ids_padded(self,
                               common_attn_metadata: CommonAttentionMetadata,
                               sampled_token_ids: torch.Tensor,
                               requests: dict[str, CachedRequestState],
                               gpu_input_batch: InputBatch,
                               discard_request_indices: torch.Tensor,
                               num_discarded_requests: int) -> \
                                tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        It also accounts for the rejected tokens in `sampled_token_ids`.
        This function must use device functions to operate on the inputs, and
        should not introduce any blocking CPU-GPU synchronization.
        """

        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array([
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                common_attn_metadata.seq_lens_cpu[i].item())
            for i in range(num_reqs)
        ])
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        max_gen_len = sampled_token_ids.shape[-1]
        device = sampled_token_ids.device

        next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = torch.empty(num_reqs, dtype=torch.int32, device=device)

        BLOCK = triton.next_power_of_2(max_gen_len)
        grid = (num_reqs, )

        prepare_next_token_ids_kernel[grid](
            sampled_token_ids,
            self.backup_next_token_ids.gpu,
            discard_request_indices,
            gpu_input_batch.vocab_size,
            max_gen_len,
            num_discarded_requests,
            num_reqs,
            next_token_ids,
            valid_sampled_tokens_count,
            BLOCK_SIZE=BLOCK,
        )

        if self.prepare_next_token_ids_padded_event is not None:
            self.runner.prev_valid_sampled_tokens_count_pinned_cpu[:valid_sampled_tokens_count.shape[0]].copy_(valid_sampled_tokens_count, non_blocking=True)
            self.runner.prev_valid_sampled_tokens_count_pinned_cpu[-1] = 1
            self.prepare_next_token_ids_padded_event.record()

            self.runner.prev_next_token_ids = next_token_ids

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        num_seqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = (query_start_loc_cpu[1:] -
                                    query_start_loc_cpu[:-1])
        total_num_tokens = query_start_loc_cpu[-1].item()

        token_indices_to_sample = torch.empty(num_seqs, dtype=torch.int, device=device)
        token_indices = torch.empty(total_num_tokens, dtype=torch.int, device=device)

        BLOCK = triton.next_power_of_2(total_num_tokens)
        grid = lambda meta: (triton.cdiv(num_seqs, meta['BLOCK_SIZE']),)
        prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_seqs,
            self.arange,
            token_indices,
            total_num_tokens,
            BLOCK_SIZE=BLOCK,
        )

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
        )

        return spec_common_attn_metadata, token_indices, token_indices_to_sample
