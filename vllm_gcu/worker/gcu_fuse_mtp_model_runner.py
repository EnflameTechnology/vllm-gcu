import gc
from typing import Optional, Any, List, Union
import time
from unittest.mock import patch
import numpy as np
import torch
from vllm.utils import cdiv
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.kv_transfer import has_kv_transfer_group
import vllm.envs as envs
from vllm.config import (CUDAGraphMode, set_current_vllm_config)

from vllm.sequence import IntermediateTensors
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (EncoderOnlyAttentionSpec)
from vllm.v1.outputs import (ModelRunnerOutput, LogprobsLists, EMPTY_MODEL_RUNNER_OUTPUT,
                             LogprobsTensors, SamplerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.ubatch_splitting import ubatch_split
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import (AsyncGPUModelRunnerOutput, PerLayerAttnMetadata)
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata, split_attn_metadata)
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm_gcu.utils import (
    set_gcu_forward_context,
    dump_memory_snapshot_when_exception,
)
from vllm_gcu.kernels.rejection_sampler import GCURejectionSampler
from vllm_gcu.utils import get_tx_ctx, topstx_wrapper
from vllm_gcu.worker.gcu_model_runner import GCUModelRunner, GCUAsyncGPUModelRunnerOutput

logger = init_logger(__name__)

class FuseMTPGCUAsyncGPUModelRunnerOutput(GCUAsyncGPUModelRunnerOutput):

    def __init__(self, delay_update_output_token_ids = False, num_output_placeholder=1,
                 req_ids = None, requests = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_update_output_token_ids = delay_update_output_token_ids
        if delay_update_output_token_ids:
            self.batch_req_ids = req_ids
            self.requests = requests
            self.num_output_placeholder = num_output_placeholder

    def get_output(self):
        max_gen_len = self._sampled_token_ids_cpu.shape[-1]
        if max_gen_len == 1:
            return super().get_output()

        self.wait_event_ready()
        del self._sampled_token_ids
        valid_sampled_token_ids = GCURejectionSampler.parse_output(
            self._sampled_token_ids_cpu,
            self.vocab_size,
        )
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()
        if self.delay_update_output_token_ids:
            for i in range(len(self.batch_req_ids)):
                if i in self._invalid_req_indices:
                    continue
                req_id = self.batch_req_ids[i]
                if req_id in self.requests:
                    req = self.requests[req_id]
                    assert len(req.output_token_ids) >= self.num_output_placeholder
                    try:
                        placeholder_idx = req.output_token_ids.index(self.vocab_size)
                        num_rejected = self.num_output_placeholder - len(valid_sampled_token_ids[i])
                        if num_rejected > 0:
                            req.output_token_ids[-num_rejected:] = []
                        req.output_token_ids[
                            placeholder_idx:placeholder_idx+len(valid_sampled_token_ids[i])
                            ] = valid_sampled_token_ids[i]
                    except:
                        logger.error(f'fused_mtp delay update output ids, placeholder not found')
        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output

class FuseMTPGCUModelRunner(GCUModelRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.expand_max_num_reqs = (self.get_spec_k() + 1) * self.max_num_reqs
        self.temperature = torch.full((self.expand_max_num_reqs,),
                                        fill_value=float('inf'),
                                        dtype=torch.float32,
                                        device=self.device)
        self.temperature_cpu_tensor = torch.full((self.expand_max_num_reqs,),
                                                    fill_value=float('inf'),
                                                    dtype=torch.float32,
                                                    device="cpu",
                                                    pin_memory=True)
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()

        self.top_p = torch.ones(self.expand_max_num_reqs,
                                    dtype=torch.float32,
                                    device=self.device)
        self.top_p_cpu_tensor = torch.ones(self.expand_max_num_reqs,
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=True)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()

        self.top_k = torch.ones(self.expand_max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        self.top_k_cpu_tensor = torch.ones(self.expand_max_num_reqs,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=True)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()

        self.draft_tokens = torch.zeros((self.max_num_reqs, self.get_spec_k()),
                                        dtype=torch.int32,
                                        device=self.device)
        self.draft_tokens_cpu_tensor = torch.zeros((self.max_num_reqs, self.get_spec_k()),
                                                    dtype=torch.int32,
                                                    device="cpu",
                                                    pin_memory=True)
        self.draft_tokens_cpu = self.draft_tokens_cpu_tensor.numpy()

        self.repetition_penalties = torch.ones(self.max_num_reqs,
                                                dtype=torch.float32,
                                                device=self.device)
        self.repetition_penalties_cpu_tensor = torch.ones(self.max_num_reqs,
                                                            dtype=torch.float32,
                                                            device="cpu",
                                                            pin_memory=True)
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()

        self.frequency_penalties = torch.zeros(self.max_num_reqs,
                                                dtype=torch.float32,
                                                device=self.device)
        self.frequency_penalties_cpu_tensor = torch.zeros(self.max_num_reqs,
                                                            dtype=torch.float32,
                                                            device="cpu",
                                                            pin_memory=True)
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.presence_penalties = torch.zeros(self.max_num_reqs,
                                                dtype=torch.float32,
                                                device=self.device)
        self.presence_penalties_cpu_tensor = torch.zeros(self.max_num_reqs,
                                                            dtype=torch.float32,
                                                            device="cpu",
                                                            pin_memory=True)
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()

        self.max_penalty_prompt_len = self.vllm_config.additional_config.get("deepseek_fused_mtp_penalty_max_prompt_len", 1)
        self.max_penalty_output_len = self.vllm_config.additional_config.get("deepseek_fused_mtp_penalty_max_output_len", 1)
        vocab_size = self.model_config.get_vocab_size()
        self.output_token_ids = torch.full((self.max_num_reqs, self.max_penalty_output_len),
                                            fill_value = vocab_size,
                                            dtype=torch.int64,
                                            device=self.device)
        self.output_token_ids_cpu_tensor = torch.full((self.max_num_reqs, self.max_penalty_output_len),
                                            fill_value = vocab_size,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=True)
        self.output_token_ids_cpu = self.output_token_ids_cpu_tensor.numpy()
        self.prompt_token_ids = torch.full((self.max_num_reqs, self.max_penalty_prompt_len),
                                            fill_value = vocab_size,
                                            dtype=torch.int64,
                                            device=self.device)
        self.prompt_token_ids_cpu_tensor = torch.full((self.max_num_reqs, self.max_penalty_prompt_len),
                                            fill_value = vocab_size,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=True)
        self.prompt_token_ids_cpu = self.prompt_token_ids_cpu_tensor.numpy()

        self.reorder_batch_threshold = 1 + self.get_spec_k()

        self.delay_update_output_token_ids = True
        
        if hasattr(self, "drafter") and self.drafter is not None:
            self.drafter = None
            delattr(self, "drafter")
        if self.use_async_scheduling:
            from vllm_gcu.patch.patch_0_11_0.block_table import (compute_slot_mapping_device,
                                                                    multi_compute_slot_mapping_device)
            from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
            patch.object(BlockTable, "compute_slot_mapping_device", compute_slot_mapping_device,
                    create=True).start()
            patch.object(MultiGroupBlockTable, "compute_slot_mapping_device", multi_compute_slot_mapping_device,
                            create=True).start()
            self.tmp_draft_token_ids = torch.zeros((self.max_num_reqs, self.get_spec_k()),
                                                    dtype = torch.int32,
                                                    device = self.device)
            self.num_rejected_tokens_cpu_tensor = torch.zeros((self.max_num_reqs,),
                                                    dtype=torch.int32,
                                                    device="cpu",
                                                    pin_memory=True)
            self.num_rejected_tokens_cpu = self.num_rejected_tokens_cpu_tensor.numpy()
            self.num_rejected_tokens = torch.zeros((self.max_num_reqs,),
                                                    dtype=torch.int32,
                                                    device=self.device)
            self.prev_num_rejected_tokens = None

    @topstx_wrapper
    def prepare_fused_mtp_input(self,
                                sampling_metadata: SamplingMetadata,
                                batch_size: int,
                                num_decodes: int,
                                num_prefills: int,
                                spec_k: int,
                                scheduled_spec_decode_tokens: dict[str, List[int]],
                                first_kv_transfer: dict[str, bool]):
        expand_cnt = spec_k + 1
        temperature_cpu = self.input_batch.temperature_cpu[:batch_size]
        top_p_cpu = self.input_batch.top_p_cpu[:batch_size]
        top_k_cpu = self.input_batch.top_k_cpu[:batch_size]
        repetition_penalties_cpu = self.input_batch.repetition_penalties_cpu[:batch_size]
        frequency_penalties_cpu = self.input_batch.frequency_penalties_cpu[:batch_size]
        presence_penalties_cpu = self.input_batch.presence_penalties_cpu[:batch_size]

        if self.vllm_config.additional_config.get("deepseek_fused_mtp_use_penalty", True):
            req_output_token_ids = self.input_batch.req_output_token_ids
            vocab_size = self.model_config.get_vocab_size()
            max_req_ot_len = max(len(req_ot) for req_ot in req_output_token_ids)
            output_token_ids_pad = np.full((len(req_output_token_ids), max_req_ot_len),
                                               vocab_size, dtype=np.int64)
            for i, req_ot in enumerate(req_output_token_ids):
                output_token_ids_pad[i, :len(req_ot)] = req_ot
            output_token_ids = output_token_ids_pad[:batch_size, -self.max_penalty_output_len:]

            if sampling_metadata.prompt_token_ids is not None:
                prompt_tokens = sampling_metadata.prompt_token_ids.cpu().numpy()
                prompt_token_ids = prompt_tokens[:batch_size, -self.max_penalty_prompt_len:]
            else:
                prompt_token_ids = np.empty((num_prefills + num_decodes, 0), dtype=np.int64)

        expand_reqs = num_decodes * expand_cnt + num_prefills

        self.temperature_cpu[:expand_reqs] = np.concatenate((temperature_cpu[:num_decodes].repeat(
            expand_cnt), temperature_cpu[num_decodes:],))

        self.top_p_cpu[:expand_reqs] = np.concatenate(
            (top_p_cpu[:num_decodes].repeat(expand_cnt), top_p_cpu[num_decodes:],))

        self.top_k_cpu[:expand_reqs] = np.concatenate(
            (top_k_cpu[:num_decodes].repeat(expand_cnt), top_k_cpu[num_decodes:],))

        self.temperature[:expand_reqs].copy_(self.temperature_cpu_tensor[:expand_reqs], non_blocking=True)
        self.top_p[:expand_reqs].copy_(self.top_p_cpu_tensor[:expand_reqs], non_blocking=True)
        self.top_k[:expand_reqs].copy_(self.top_k_cpu_tensor[:expand_reqs], non_blocking=True)

        if self.vllm_config.additional_config.get("deepseek_fused_mtp_use_penalty", True):
            num_reqs = self.input_batch.num_reqs
            self.repetition_penalties_cpu[:num_reqs] = np.concatenate((repetition_penalties_cpu[:num_decodes],
                                                                            repetition_penalties_cpu[num_decodes:]))
            self.repetition_penalties[:num_reqs].copy_(self.repetition_penalties_cpu_tensor[:num_reqs], non_blocking=True)

            self.frequency_penalties_cpu[:num_reqs] = np.concatenate((frequency_penalties_cpu[:num_decodes],
                                                                            frequency_penalties_cpu[num_decodes:]))
            self.frequency_penalties[:num_reqs].copy_(self.frequency_penalties_cpu_tensor[:num_reqs], non_blocking=True)
            self.presence_penalties_cpu[:num_reqs] = np.concatenate((presence_penalties_cpu[:num_decodes],
                                                                           presence_penalties_cpu[num_decodes:]))
            self.presence_penalties[:num_reqs].copy_(self.presence_penalties_cpu_tensor[:num_reqs], non_blocking=True)
            self.output_token_ids_cpu[:num_reqs,:output_token_ids.shape[1]] = np.concatenate((output_token_ids[:num_decodes],
                                                                      output_token_ids[num_decodes:]),axis=0)
            self.output_token_ids[:num_reqs].copy_(self.output_token_ids_cpu_tensor[:num_reqs], non_blocking=True)
            self.prompt_token_ids_cpu[:num_reqs,:prompt_token_ids.shape[1]] = np.concatenate((prompt_token_ids[:num_decodes],
                                                                      prompt_token_ids[num_decodes:]),axis=0)
            self.prompt_token_ids[:num_reqs].copy_(self.prompt_token_ids_cpu_tensor[:num_reqs], non_blocking=True)            


        for i, req_id in enumerate(self.input_batch.req_ids[:num_decodes]):
            if req_id in first_kv_transfer and req_id not in scheduled_spec_decode_tokens:
                self.draft_tokens_cpu[i, :] = [0] * spec_k
            else:
                self.draft_tokens_cpu[i, :] = [x if x>=0 else 0 for x in scheduled_spec_decode_tokens[req_id]]

        self.draft_tokens[:batch_size].copy_(self.draft_tokens_cpu_tensor[:batch_size, :spec_k], non_blocking=True)

        if self.use_async_scheduling and self._draft_token_ids is not None:
            prev_req_id_to_index = self.input_batch.prev_req_id_to_index
            assert prev_req_id_to_index is not None
            prev_common_req_indices = []
            update_indices = []
            for req_id, cur_index in self.input_batch.req_id_to_index.items():
                if (prev_index := prev_req_id_to_index.get(req_id)) is not None \
                    and prev_index < self._draft_token_ids.shape[0]:
                    prev_common_req_indices.append(prev_index)
                    update_indices.append(cur_index)
            if len(prev_common_req_indices) > 0:
                update_indices_tensor = torch.tensor(update_indices,
                                              dtype=torch.int64,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)
                prev_common_req_indices_tensor = \
                    torch.tensor(prev_common_req_indices,
                                dtype=torch.int64,
                                pin_memory=self.pin_memory).to(
                                    self.device,
                                    non_blocking=True
                                )
                self.draft_tokens[update_indices_tensor, :] = \
                    self._draft_token_ids[prev_common_req_indices_tensor, :]

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            # NOTE(lucas): currently no backend supports the custom masking
            #  required for DCP with q_len > 1, so we assert here. Remove this
            #  assert once the custom mask is support is added to FA3.
            if self.dcp_world_size > 1:
                assert self.reorder_batch_threshold == 1, \
                    "DCP not support reorder_batch_threshold > 1 now."
            self.reorder_batch_fused_mtp(self.input_batch,
                                            scheduler_output,
                                            decode_threshold=self.reorder_batch_threshold,
                                            requests = self.requests)

    def reorder_batch_fused_mtp(
        self,
        input_batch: "InputBatch",
        scheduler_output: "SchedulerOutput",
        decode_threshold: int = 1,
        requests: "CachedRequestState" = None,
    ) -> bool:
        """
        Reorders the batch to split into prefill and decode requests; places all
        requests with <= decode_threshold tokens at the front of the batch.

        Returns:
            True if the batch was modified, False otherwise.
        Notice: one key difference with reorder_batch_to_split_decodes_and_prefills is that
                this implementation keeps decode/prefill semantics in best effort.
                why official implementation doesn't need to keep such original semantics is that
                decode/prefill semantics only matters in attention computation efficiency.
                however, in deepseek-fused-mtp, when prefill semantics for a real prefill is loss,
                acceptance results and target token ids for prefill will be mistaken.
        """
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the back using the least
        # amount of swaps possible. (NOTE for now we loosely use "decode" to mean
        # requests where attention is likely memory-bound and "prefill" to mean
        # requests where attention is likely compute-bound, TODO(lucas): figure out
        # a better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0
        assert self.vllm_config.speculative_config is not None
        spec_k = self.vllm_config.speculative_config.num_speculative_tokens
        assert decode_threshold == spec_k + 1
        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_computed_tokens = requests[req_id].num_computed_tokens
            prompt_token_ids = requests[req_id].prompt_token_ids
            if num_computed_tokens + 1 < len(prompt_token_ids):
                if req_id in scheduler_output.first_transfer_request:
                    decodes.append(i)
                    num_decode_tokens += num_tokens
                else:
                    prefills.append(i)
                    num_prefill_tokens += num_tokens
            else:
                if num_tokens == decode_threshold:
                    decodes.append(i)
                    num_decode_tokens += num_tokens
                else:
                    prefills.append(i)
                    num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            decode_idx = decodes[num_decodes - i]
            if decode_idx < num_decodes:
                break

            input_batch.swap_states(prefills[i - 1], decode_idx)
            modified_batch = True
        # make sure attention builder split logic align to this logic
        # assert len(self.attn_groups[0]) == 1
        attn_builder = self.attn_groups[0][0].get_metadata_builder()
        attn_builder._num_decodes = num_decodes
        attn_builder._num_prefills = num_prefills
        attn_builder._num_decode_tokens = num_decode_tokens
        attn_builder._num_prefill_tokens = num_prefill_tokens
        attn_builder.reorder_batch_threshold = 1 + spec_k
        return modified_batch

    def calculate_reorder_batch_threshold(self) -> None:
        assert self.vllm_config.speculative_config is not None
        spec_k = self.vllm_config.speculative_config.num_speculative_tokens
        return spec_k + 1

    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[PerLayerAttnMetadata, torch.Tensor,
               Optional[SpecDecodeMetadata], np.ndarray,
               Optional[CommonAttentionMetadata], int, Optional[UBatchSlices],
               Optional[torch.Tensor]]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        # But num_computed_tokens_cpu is biased, we will fix this value in following steps.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        #for pure decode, we do not need to fill input_ids gpu buffer from input_batch.input_token_ids
        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           token_indices_tensor,
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[:num_reqs + 1]

        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        num_tokens_padded = num_tokens_unpadded + self.get_local_padding(
            num_tokens_unpadded)
        uniform_decode = \
            (max_num_scheduled_tokens == self.uniform_decode_query_len) and \
            (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
        ubatch_slices, num_tokens_after_padding = \
            ubatch_split(num_scheduled_tokens,
                         num_tokens_unpadded,
                         num_tokens_padded,
                         uniform_decode=uniform_decode,
                         vllm_config=self.vllm_config)

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()
        seq_lens = self.seq_lens.gpu[:num_reqs]
        max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [
            self.requests[r].num_prompt_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np
        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)
        # save this useless copy
        #self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)

        # Common case (1D positions)
        self.positions.copy_to_gpu(total_num_scheduled_tokens)

        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for req_id, draft_token_ids in (
                scheduler_output.scheduled_spec_decode_tokens.items()):
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)
        
        # spec_decode_metadata = self._calc_spec_decode_metadata(
        #    num_draft_tokens, cu_num_tokens)
        spec_decode_metadata = None
        #logits_indices = spec_decode_metadata.logits_indices
        # spec_decode_metadata = None
        logits_indices = self._calc_logist_index(
            num_draft_tokens, cu_num_tokens)


        logits_indices_padded = None
        if self.cache_config.kv_sharing_fast_prefill:
            logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices)

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        # Used in the below loop.
        query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        spec_decode_common_attn_metadata = None

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            encoder_seq_lens = self._get_encoder_seq_lens(
                scheduler_output, kv_cache_group_spec.kv_cache_spec, num_reqs)

            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
                num_common_prefix_blocks = 0
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs)
                slot_mapping = blk_table.slot_mapping.gpu[:
                                                          total_num_scheduled_tokens]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode.
                blk_table.slot_mapping.gpu[total_num_scheduled_tokens:].fill_(
                    -1)
                num_common_prefix_blocks = (
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id])

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                max_seq_len=max_seq_len,
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
                logits_indices_padded=logits_indices_padded,
                num_logits_indices=(self.get_spec_k() + 1) * self.input_batch.num_reqs, #logits_indices.size(0),
                causal=True,
                encoder_seq_lens=encoder_seq_lens,
            )

            if (self.speculative_config
                    and spec_decode_common_attn_metadata is None):
                if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
                    if (self.drafter.attn_layer_names[0]
                            in kv_cache_group_spec.layer_names):
                        spec_decode_common_attn_metadata = common_attn_metadata
                else:
                    spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                builder = attn_group.get_metadata_builder()
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_common_prefix_blocks,
                        attn_group.kv_cache_spec,
                        builder,
                    )

                extra_attn_metadata_args = {}

                if ubatch_slices is not None:
                    assert not self.vllm_config.additional_config["deepseek_fused_mtp"], \
                        "deepseek fused mtp is not ready for dbo"
                    common_attn_metadata_list = split_attn_metadata(
                        ubatch_slices, common_attn_metadata)
                    for ubid, common_attn_metadata in enumerate(
                            common_attn_metadata_list):
                        attn_metadata_i = (attn_group.get_metadata_builder(
                            ubatch_id=ubid).build(
                                common_prefix_len=common_prefix_len,
                                common_attn_metadata=common_attn_metadata))
                        for layer_name in kv_cache_group_spec.layer_names:
                            assert type(attn_metadata) is list
                            attn_metadata[ubid][layer_name] = attn_metadata_i
                else:
                    assert isinstance(attn_metadata, dict)
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                    for layer_name in attn_group.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

                    if self.vllm_config.additional_config["deepseek_fused_mtp"]:
                        if "ds_main_with_mtp" not in attn_metadata:
                            attn_metadata["ds_main_with_mtp"] = attn_metadata_i

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        if self.positions_tensor is not None:
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            self.positions.gpu[:total_num_scheduled_tokens].copy_(self.positions_tensor)
            self.positions_tensor = None

        return (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens, spec_decode_common_attn_metadata,
                max_num_scheduled_tokens, ubatch_slices,
                num_tokens_after_padding)

    def _prepare_model_runner_output(self, num_nans_in_logits, logprobs_lists, valid_sampled_token_ids, prompt_logprobs_dict, 
                        req_ids_output_copy, req_id_to_index_output_copy, invalid_req_indices, extra_args):

        kv_connector_output = extra_args['kv_connector_output']
        sampler_output = extra_args['sampler_output']

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        if not self.use_async_scheduling:
            return output
        
        async_output = FuseMTPGCUAsyncGPUModelRunnerOutput(
            delay_update_output_token_ids=self.delay_update_output_token_ids,
            num_output_placeholder=1 + self.get_spec_k(),
            req_ids=self.input_batch.req_ids.copy(),
            requests=self.requests,
            vocab_size=self.input_batch.vocab_size,
            event_poll_span_ms=1,
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )
        # Save ref of sampled_token_ids CPU tensor if the batch contains
        # any requests with sampling params that that require output ids.
        self.input_batch.set_async_sampled_token_ids(
            async_output._sampled_token_ids_cpu,
            async_output._async_copy_ready_event,
        )

        return async_output


    @torch.inference_mode()
    @dump_memory_snapshot_when_exception('step')
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
        """
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if num_tokens == 0:
            num_reqs = 1
            num_scheduled_tokens_list = []
        elif create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = num_tokens // 2
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [
                num_prefill_tokens
            ]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            assert num_reqs <= max_num_reqs, \
                "Do not capture num_reqs > max_num_reqs for uniform batch"
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        # assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())

        ubatch_slices = None
        num_tokens_after_padding = None

        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        if self.parallel_config.enable_dbo and allow_microbatching:
            ubatch_slices, ubatch_num_tokens_after_padding = ubatch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            # Currently when DBO is enabled `ubatch_split` returns
            # the num_tokens_after_padding for a single ubatch, but we have 2
            # TODO(sage,lucas): this is cruft that should be addressed in the
            # padding refactor.
            if ubatch_num_tokens_after_padding is not None:
                num_tokens_after_padding = ubatch_num_tokens_after_padding * 2

        # If we failed to microbatch, currently need to resynchronize
        # TODO(lucas,sage): we should be able to avoid this second sync by
        #  refactoring `get_dp_padding_ubatch` and `get_dp_padding` into
        #  a single `coordinate_batch_across_dp` function.
        if num_tokens_after_padding is None:
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
            num_tokens_after_padding = num_tokens + num_pad
            num_tokens += num_pad
        else:
            num_tokens_across_dp = num_tokens_after_padding
            num_tokens_after_padding = int(num_tokens_after_padding[0].item())

        attn_metadata: Optional[dict[str, Any]] = None
        spec_decode_common_attn_metadata = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        logits_indices = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            assert not is_profile, "profile_run must run under non-graph"
            attn_metadata = {}
            if ubatch_slices is not None:
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]

            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(
                num_scheduled_tokens) if num_tokens != 0 else (np.array([0]), None)
            self.query_start_loc.np[1:num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()
            logits_indices[:] = self.query_start_loc.gpu[1:num_reqs + 1] - 1
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    max_seq_len=self.max_model_len,
                    block_table_tensor=self.input_batch.
                    block_table[kv_cache_group_id].get_device_tensor(num_reqs),
                    slot_mapping=self.input_batch.block_table[
                        kv_cache_group_id].slot_mapping.gpu[:num_tokens],
                    causal=True)

                if (self.speculative_config
                        and spec_decode_common_attn_metadata is None):
                    if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
                        if (self.drafter.attn_layer_names[0]
                                in kv_cache_group_spec.layer_names):
                            spec_decode_common_attn_metadata = common_attn_metadata
                    else:
                        spec_decode_common_attn_metadata = common_attn_metadata

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    if ubatch_slices is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            ubatch_slices, common_attn_metadata)
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = (attn_group\
                                               .get_metadata_builder(ubatch_id=ubid)\
                                               .build_for_cudagraph_capture(common_attn_metadata))
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        assert type(attn_metadata) is dict
                        attn_metadata_i = attn_group.get_metadata_builder()\
                            .build_for_cudagraph_capture(common_attn_metadata)
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i
                        if self.vllm_config.additional_config["deepseek_fused_mtp"]:
                            if "ds_main_with_mtp" not in attn_metadata:
                                attn_metadata["ds_main_with_mtp"] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens, remove_lora):
            model_kwargs = self._init_model_kwargs(num_tokens)
            if (self.supports_mm_inputs
                    and not self.model_config.is_encoder_decoder):
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = self._init_model_kwargs(num_tokens)
            else:
                input_ids = self.input_ids.gpu[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens]
            else:
                positions = self.positions.gpu[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)

            # filter out the valid batch descriptor
            _cg_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                BatchDescriptor(num_tokens=num_tokens_after_padding,
                                uniform_decode=uniform_decode)) \
                if not is_profile else (CUDAGraphMode.NONE, None)
            if cudagraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for cudagraph capture
                assert cudagraph_runtime_mode == CUDAGraphMode.NONE or \
                    cudagraph_runtime_mode == _cg_mode, (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}.")
            else:
                cudagraph_runtime_mode = _cg_mode

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            with self.maybe_randomize_inputs(
                    input_ids), \
                    set_current_vllm_config(self.vllm_config), \
                    set_gcu_forward_context(
                        attn_metadata,
                        self.vllm_config,
                        num_tokens=num_tokens_after_padding,
                        num_tokens_across_dp=num_tokens_across_dp,
                        cudagraph_runtime_mode=cudagraph_runtime_mode,
                        batch_descriptor=batch_descriptor,
                        ubatch_slices=ubatch_slices,
                        is_dummy=True):
                if not is_profile:
                    assert num_tokens == num_tokens_after_padding
                temperature = self.temperature[:num_tokens]
                top_p = self.top_p[:num_tokens]
                top_k = self.top_k[:num_tokens]
                repetition_penalty = self.repetition_penalties[:num_tokens]
                presence_penalty = self.presence_penalties[:num_tokens]
                frequency_penalty = self.frequency_penalties[:num_tokens]
                prompt_token_ids = self.prompt_token_ids[:num_tokens]
                output_token_ids = self.output_token_ids[:num_tokens]
                draft_tokens = self.draft_tokens[:num_tokens // (self.get_spec_k() + 1), :]
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    draft_tokens=draft_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=output_token_ids,
                    logits_indices=logits_indices
                )
                hidden_states = torch.zeros((num_tokens, self.hidden_size), dtype = self.dtype,
                                                device = self.device)
                last_hidden_states = hidden_states[-1:, :] if num_tokens > 0 else \
                                        torch.empty((1, self.hidden_size), dtype = self.dtype,
                                                    device = self.device)
                if not skip_eplb:
                    self.eplb_step(is_dummy=True, is_profile=is_profile)
                return hidden_states, last_hidden_states

    def _model_forward(self, input_ids, input_positions, intermediate_tensors, inputs_embeds, model_kwargs, extra_args):

        scheduler_output = extra_args['scheduler_output']
        attn_metadata_builder = self.attn_groups[0][0].get_metadata_builder()
        num_decodes = attn_metadata_builder._num_decodes
        num_prefills = attn_metadata_builder._num_prefills
        spec_k = self.get_spec_k()
        batch_size = self.input_batch.num_reqs
        expand_reqs = num_decodes * (spec_k + 1) + num_prefills
        self.prepare_fused_mtp_input(self.input_batch.sampling_metadata,
                                        batch_size,
                                        num_decodes=num_decodes,
                                        num_prefills=num_prefills,
                                        spec_k=spec_k,
                                        scheduled_spec_decode_tokens=scheduler_output.scheduled_spec_decode_tokens,
                                        first_kv_transfer = scheduler_output.first_transfer_request)

        model_output = self.model(
            input_ids=input_ids,
            positions=input_positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            logits_indices=extra_args['logits_indices'],
            draft_tokens=self.draft_tokens[:batch_size, :spec_k],
            top_p=self.top_p[:expand_reqs],
            top_k=self.top_k[:expand_reqs],
            temperature=self.temperature[:expand_reqs],
            repetition_penalty=self.repetition_penalties[:batch_size],
            frequency_penalty=self.frequency_penalties[:batch_size],
            presence_penalty=self.presence_penalties[:batch_size],
            prompt_token_ids=self.prompt_token_ids[:batch_size],
            output_token_ids=self.output_token_ids[:batch_size],
        )
        sampled_token_ids = model_output["accepted_tokens"]
        accepted_lens = model_output["accepted_lens"]
        sampled_token_ids = sampled_token_ids[:num_decodes+num_prefills]
        accepted_lens = accepted_lens[:num_decodes+num_prefills]
        # avoid modified model_output inplace
        intermediate_result = IntermediateTensors({
            "main_model_sampled_tokens" : model_output["main_model_sampled_tokens"],
            "accepted_tokens": sampled_token_ids,
            "accepted_lens": accepted_lens,
            "next_draft_tokens": model_output["next_draft_tokens"],
            "next_token_ids": model_output["next_token_ids"],
            "hidden_states": model_output['hidden_states'],
            "logist": model_output['logist'],
        })

        extra_args['model_output'] = intermediate_result

        return model_output['hidden_states'], None

    def _determine_batch_descriptor(self, extra_args):
        
        attn_metadata_builder = self.attn_groups[0][0].get_metadata_builder()
        num_prefills = attn_metadata_builder._num_prefills
        max_query_len = extra_args['max_query_len']
        uniform_decode = (max_query_len
                            == self.uniform_decode_query_len) and (
                                extra_args['num_scheduled_tokens']
                                == self.input_batch.num_reqs * extra_args['max_query_len'])

        uniform_decode = (uniform_decode and num_prefills == 0)

        batch_descriptor = BatchDescriptor(num_tokens=extra_args['num_input_tokens'],
                                            uniform_decode=uniform_decode)

        return batch_descriptor

    def _compute_logist(self, hidden_states, extra_args):
        model_output = extra_args['model_output']
        return model_output["logist"]

    def _compute_sampler_output(self, logist, spec_decode_metadata, extra_args):
        model_output = extra_args['model_output']
        sampled_token_ids = model_output["accepted_tokens"]

        logprobs_tensor = None

        sampler_output = SamplerOutput(sampled_token_ids=sampled_token_ids,
                            logprobs_tensors=logprobs_tensor)

        return sampler_output

    def _compute_spect_tokens(self, hidden_states, aux_hidden_states, sampler_output, extra_args):
        model_output = extra_args['model_output']
        
        attn_metadata_builder = self.attn_groups[0][0].get_metadata_builder()
        num_decodes = attn_metadata_builder._num_decodes
        num_prefills = attn_metadata_builder._num_prefills

        self._draft_token_ids = model_output["next_draft_tokens"][:num_decodes+num_prefills]
        if self.use_async_scheduling:
            if self._draft_token_ids.data_ptr() == self.draft_tokens.data_ptr():
                self.tmp_draft_token_ids[:num_decodes,:].copy_(self._draft_token_ids, non_blocking = True)
                self._draft_token_ids = self.tmp_draft_token_ids[:num_decodes,:]
            valid_sampled_tokens_count = model_output["accepted_lens"]
            next_token_ids = model_output["next_token_ids"][:num_decodes+num_prefills]

            self.input_batch.prev_next_token_ids = next_token_ids.squeeze(1)
            num_draft_tokens = [self.get_spec_k() + 1] * self.input_batch.num_reqs
            if num_prefills > 0:
                num_draft_tokens[num_decodes:] = [1] * num_prefills

            self.input_batch.prev_valid_sampled_tokens_count = valid_sampled_tokens_count
            self.input_batch.prev_num_sampled_tokens = torch.tensor(
                num_draft_tokens,
                dtype=torch.int32,
                pin_memory=self.pin_memory).to(self.device, non_blocking=True)

    def _calc_logist_index(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        num_sampled_tokens = num_draft_tokens + 1

        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32)
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        logits_indices += arange
        
        logits_indices = torch.from_numpy(logits_indices).pin_memory().to(self.device,
                                                             non_blocking=True)

        return logits_indices

    @torch.inference_mode()
    @dump_memory_snapshot_when_exception('step')
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"), get_tx_ctx("Preprocess", "green", "VLLM"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config)

                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs")

                # Prepare the decoder inputs.
                (attn_metadata, logits_indices, spec_decode_metadata,
                 num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                 max_query_len, ubatch_slices, num_tokens_after_padding
                 ) = self._prepare_inputs(scheduler_output)

            extra_args = dict({})

            (
                num_scheduled_tokens,
                num_input_tokens,
                num_tokens_across_dp,
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
            ) = self._preprocess(scheduler_output, intermediate_tensors,
                                 ubatch_slices, num_tokens_after_padding)

        extra_args = {
            'scheduler_output':scheduler_output,
            'attn_metadata':attn_metadata,
            'num_input_tokens':num_input_tokens,
            'num_scheduled_tokens':num_scheduled_tokens,
            'logits_indices':logits_indices,
            'spec_decode_metadata':spec_decode_metadata,
            'num_scheduled_tokens_np':num_scheduled_tokens_np,
            'max_query_len':max_query_len,
            "spec_decode_common_attn_metadata":spec_decode_common_attn_metadata
        }

        batch_descriptor = self._determine_batch_descriptor(extra_args)
        cudagraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(batch_descriptor)

        # This is currently to get around the assert in the DPMetadata
        # where it wants `num_tokens_across_dp` to align with `num_tokens`
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens
            extra_args.update({'num_input_tokens':num_input_tokens})

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (set_current_vllm_config(self.vllm_config),
              set_gcu_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices
            ), record_function_or_nullcontext("Forward"), get_tx_ctx("Forward", "green", "VLLM"),
              self.maybe_get_kv_connector_output(scheduler_output) as
              kv_connector_output):
                hidden_states, aux_hidden_states = self._model_forward(input_ids, positions, intermediate_tensors, inputs_embeds, model_kwargs, extra_args)
        
        extra_args.update({"kv_connector_output":kv_connector_output})

        if not self.broadcast_pp_output:
            if not get_pp_group().is_last_rank:
                # Return the intermediate tensors.
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states
            if self.is_pooling_model:
                # Return the pooling output.
                output = self._pool(hidden_states, num_scheduled_tokens,
                                    num_scheduled_tokens_np)
                output.kv_connector_output = kv_connector_output
                return output

        logist = self._compute_logist(hidden_states, extra_args)
        
        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            apply_grammar_bitmask(scheduler_output, self.input_batch,
                                logist, self.device)


        sampler_output = self._compute_sampler_output(logist, spec_decode_metadata, extra_args)

        with record_function_or_nullcontext("Bookkeep"), get_tx_ctx("Bookkeep", "green", "VLLM"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices
            ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                            logist, # logits
                            hidden_states, # hidden_states
                            num_scheduled_tokens)

        extra_args.update({
            "sampler_output":sampler_output,
            "valid_sampled_token_ids":valid_sampled_token_ids
        })
        
        self._compute_spect_tokens(hidden_states, aux_hidden_states, sampler_output, extra_args)

        with record_function_or_nullcontext("EPLB"), get_tx_ctx("EPLB", "green", "VLLM"):
            self.eplb_step()

        output = self._prepare_model_runner_output(num_nans_in_logits, logprobs_lists, valid_sampled_token_ids, 
                        prompt_logprobs_dict, req_ids_output_copy, req_id_to_index_output_copy, invalid_req_indices, extra_args)

        return output

    @topstx_wrapper
    def _bookkeeping_sync(
        self, scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput, logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor, num_scheduled_tokens: int
    ) -> tuple[
            dict[str, int],
            Optional[LogprobsLists],
            list[list[int]],
            dict[str, Optional[LogprobsTensors]],
            list[str],
            dict[str, int],
            list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        discard_sampled_tokens_req_indices = \
            self.discard_request_indices.np[:self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = \
            self.input_batch.req_id_to_index.copy()

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        invalid_req_indices = []
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)].clear()
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_sampled_token_ids_invalid_indices = invalid_req_indices_set
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }
            scheduled_num_spec_tokens = [
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            ]
            self.input_batch.prev_num_tokens_to_verify = torch.tensor(scheduled_num_spec_tokens, dtype=torch.int32, pin_memory=self.pin_memory).to(self.device, non_blocking=True)

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if \
                    req_idx not in invalid_req_indices_set else None
                if self.vllm_config.additional_config["deepseek_fused_mtp"]:
                    num_new_tokens = 1 + self.get_spec_k()
                    new_output_token_ids = [self.input_batch.vocab_size] * num_new_tokens if \
                        req_idx not in invalid_req_indices_set else None
                else:
                    new_output_token_ids = sampled_ids
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]
                new_output_token_ids = sampled_ids
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            
            #for fused_mtp with async-scheduling, new_output_token_ids is with extra spec_k placeholders
            req_state.output_token_ids.extend(new_output_token_ids)

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )
