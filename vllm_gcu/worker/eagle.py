#!/usr/bin/env python
# coding=utf-8
from typing import Optional
import copy
import torch

from vllm.config import VllmConfig, CUDAGraphMode
from vllm.utils import cdiv
from vllm.forward_context import BatchDescriptor
from vllm.v1.spec_decode.eagle import EagleProposer, PADDING_SLOT_ID
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm_gcu.utils import set_gcu_forward_context


class EagleProposerWithGraph(EagleProposer):

    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 runner=None):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        self.runtime_mode = cudagraph_mode.decode_mode()
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

        if self.runtime_mode == CUDAGraphMode.NONE:
            self.model1 = self.model2 = self.model
        else:
            self.model1 = CUDAGraphWrapper(self.model,
                                           self.vllm_config,
                                           runtime_mode=self.runtime_mode)
            self.model2 = CUDAGraphWrapper(self.model,
                                           self.mtp_config,
                                           runtime_mode=self.runtime_mode)

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
            assert isinstance(self.model1, Eagle3LlamaForCausalLM)
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

        if num_tokens <= self.cudagraph_batch_sizes[
                -1] and num_tokens == batch_size * (
                    self.num_speculative_tokens + 1):
            cudagraph_runtime_mode = self.runtime_mode
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE

        batch_descriptor = None
        num_input_tokens = num_tokens
        if cudagraph_runtime_mode != CUDAGraphMode.NONE:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)

            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=True)
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher1.dispatch(batch_descriptor))
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

        if (self.runtime_mode != CUDAGraphMode.NONE
                and batch_size <= self.mtp_cudagraph_batch_sizes[-1]):
            cudagraph_runtime_mode = self.runtime_mode
            input_batch_size = self.mtp_config.pad_for_cudagraph(batch_size)
            batch_descriptor = BatchDescriptor(num_tokens=input_batch_size,
                                               uniform_decode=True)
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher2.dispatch(batch_descriptor))
        else:
            input_batch_size = batch_size
            batch_descriptor = None
            cudagraph_runtime_mode = CUDAGraphMode.NONE

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

        if cudagraph_runtime_mode != CUDAGraphMode.NONE:
            cudagraph_runtime_mode = (
                self.runtime_mode if cudagraph_runtime_mode
                == self.runtime_mode else CUDAGraphMode.NONE)

        if common_attn_metadata is not None:
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
        cudagraph_runtime_mode1 = cudagraph_runtime_mode
        if num_tokens > self.cudagraph_batch_sizes[-1]:
            cudagraph_runtime_mode1 = CUDAGraphMode.NONE

        per_layer_attn_metadata = None
        batch_descriptor = None
        num_input_tokens = num_tokens

        if cudagraph_runtime_mode1 != CUDAGraphMode.NONE:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)

            assert spec_common_attn_metadata is not None
            per_layer_attn_metadata = {}
            if self.attn_metadata_builder is None:
                attn_metadata_builder = self._get_attention_metadata_builder()
            else:
                attn_metadata_builder = self.attn_metadata_builder
            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=spec_common_attn_metadata, draft_index=0)
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            uniform_decode = cudagraph_runtime_mode1 == CUDAGraphMode.FULL
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=uniform_decode)
            cudagraph_runtime_mode1, batch_descriptor = (
                self.cudagraph_dispatcher1.dispatch(batch_descriptor))

        with set_gcu_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode1,
                batch_descriptor=batch_descriptor,
        ):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
            else:
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None

            self.model1(
                input_ids=input_ids,
                positions=self.positions[:num_input_tokens],
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )

        if self.num_speculative_tokens == 1:
            return

        batch_size = cdiv(num_tokens, (self.num_speculative_tokens + 1))

        # mtp>1
        cudagraph_runtime_mode2 = cudagraph_runtime_mode
        if batch_size > self.mtp_cudagraph_batch_sizes[-1]:
            cudagraph_runtime_mode2 = CUDAGraphMode.NONE

        per_layer_attn_metadata = None
        batch_descriptor = None
        input_batch_size = batch_size

        if cudagraph_runtime_mode2 != CUDAGraphMode.NONE:
            input_batch_size = self.mtp_config.pad_for_cudagraph(batch_size)

            assert spec_common_attn_metadata is not None
            per_layer_attn_metadata = {}

            spec_common_attn_metadata.num_actual_tokens = batch_size
            spec_common_attn_metadata.max_query_len = 1
            spec_common_attn_metadata.query_start_loc = self.arange[:batch_size + 1]
            spec_common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[:batch_size + 1]).clone()
            spec_common_attn_metadata.slot_mapping=self.slot_mapping[:batch_size]
            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=spec_common_attn_metadata, draft_index=1)
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            uniform_decode = cudagraph_runtime_mode2 == CUDAGraphMode.FULL
            batch_descriptor = BatchDescriptor(num_tokens=input_batch_size,
                                               uniform_decode=uniform_decode)
            cudagraph_runtime_mode2, batch_descriptor = (
                self.cudagraph_dispatcher2.dispatch(batch_descriptor))

        for token_index in range(self.num_speculative_tokens - 1):
            with set_gcu_forward_context(
                    per_layer_attn_metadata,
                    self.mtp_config,
                    num_tokens=input_batch_size,
                    cudagraph_runtime_mode=cudagraph_runtime_mode2,
                    batch_descriptor=batch_descriptor,
            ):
                if self.is_multimodal_model:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:input_batch_size]
                else:
                    input_ids = self.input_ids[:input_batch_size]
                    inputs_embeds = None

                self.model2(
                    input_ids=input_ids,
                    positions=self.positions[:input_batch_size],
                    hidden_states=self.hidden_states[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
