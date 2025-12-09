import gc
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Union, Any, cast, List
import time
from unittest.mock import patch
import numpy as np
import torch
from vllm.utils import cdiv
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.distributed.parallel_state import (get_tp_group, get_pp_group, get_ep_group,
                                             prepare_communication_buffer_for_model, graph_capture)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
import vllm.envs as envs
from vllm.config import (CUDAGraphMode, set_current_vllm_config, get_layers_from_vllm_config)
from vllm.sampling_params import SamplingType
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sequence import IntermediateTensors
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import (KVCacheConfig, EncoderOnlyAttentionSpec)
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.outputs import (ModelRunnerOutput, EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds,
                             LogprobsLists, LogprobsTensors, SamplerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.worker.ubatch_splitting import ubatch_split
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import (GPUModelRunner, AsyncGPUModelRunnerOutput, PerLayerAttnMetadata)
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata, split_attn_metadata,
                                              reorder_batch_to_split_decodes_and_prefills)
from vllm_gcu.utils import (
    set_gcu_forward_context,
    dump_memory_snapshot_when_exception,
    prepare_communication_buffer_for_model_noep,
)
from vllm_gcu.compilation.pass_manager import PassManager, SingletonPostGradPassManager
from vllm_gcu.kernels.sampler import GCUSampler
from vllm_gcu.kernels.rejection_sampler import GCURejectionSampler
from vllm_gcu.utils import get_tx_ctx, topstx_wrapper

logger = init_logger(__name__)

class GCUAsyncGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):

    def __init__(self, vocab_size: int, event_poll_span_ms = -1, 
                 delay_update_output_token_ids = False, num_output_placeholder=1,
                 req_ids = None, requests = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.event_poll_span_ms = event_poll_span_ms
        self.delay_update_output_token_ids = delay_update_output_token_ids
        if delay_update_output_token_ids:
            self.batch_req_ids = req_ids
            self.requests = requests
            self.num_output_placeholder = num_output_placeholder

    def wait_event_ready(self):
        if self.event_poll_span_ms > 0:
            while not self._async_copy_ready_event.query():
                time.sleep(self.event_poll_span_ms / 1000)
        else:
            self._async_copy_ready_event.synchronize()

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

class GCUModelRunner(GPUModelRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prev_valid_sampled_tokens_count_pinned_cpu = torch.zeros(
            (self.max_num_reqs + 1,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory)
        self.prev_valid_sampled_tokens_count_pinned_cpu[-1] = 0
        self.prev_next_token_ids: torch.Tensor | None = None
        self.prepare_next_token_ids_padded_event: torch.cuda.Event | None = None
        if self.speculative_config and self.use_async_scheduling and \
            not self.vllm_config.additional_config["deepseek_fused_mtp"]:
            self.prepare_next_token_ids_padded_event = torch.cuda.Event()
            self.prepare_next_token_ids_padded_event.record(
                torch.cuda.default_stream())

        logprobs_mode = self.sampler.topk_topp_sampler.logprobs_mode
        self.sampler = GCUSampler(logprobs_mode)
        if hasattr(self, "rejection_sampler"):
            self.rejection_sampler = GCURejectionSampler()
        if self.vllm_config.additional_config["deepseek_fused_mtp"]:
            assert self.compilation_config.full_cuda_graph or \
                self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL, \
                "deepseek with fused mtp requires full cuda graph"
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

            self.repetition_penalties = torch.ones(self.expand_max_num_reqs,
                                                   dtype=torch.float32,
                                                   device=self.device)
            self.repetition_penalties_cpu_tensor = torch.ones(self.expand_max_num_reqs,
                                                              dtype=torch.float32,
                                                              device="cpu",
                                                              pin_memory=True)
            self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()

            self.frequency_penalties = torch.zeros(self.expand_max_num_reqs,
                                                   dtype=torch.float32,
                                                   device=self.device)
            self.frequency_penalties_cpu_tensor = torch.zeros(self.expand_max_num_reqs,
                                                              dtype=torch.float32,
                                                              device="cpu",
                                                              pin_memory=True)
            self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
            self.presence_penalties = torch.zeros(self.expand_max_num_reqs,
                                                   dtype=torch.float32,
                                                   device=self.device)
            self.presence_penalties_cpu_tensor = torch.zeros(self.expand_max_num_reqs,
                                                              dtype=torch.float32,
                                                              device="cpu",
                                                              pin_memory=True)
            self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()

            self.max_penalty_prompt_len = self.vllm_config.additional_config.get("deepseek_fused_mtp_penalty_max_prompt_len", 1)
            self.max_penalty_output_len = self.vllm_config.additional_config.get("deepseek_fused_mtp_penalty_max_output_len", 1)
            vocab_size = self.model_config.get_vocab_size()
            self.output_token_ids = torch.full((self.expand_max_num_reqs, self.max_penalty_output_len),
                                               fill_value = vocab_size,
                                               dtype=torch.int64,
                                               device=self.device)
            self.output_token_ids_cpu_tensor = torch.full((self.expand_max_num_reqs, self.max_penalty_output_len),
                                               fill_value = vocab_size,
                                               dtype=torch.int64,
                                               device="cpu",
                                               pin_memory=True)
            self.output_token_ids_cpu = self.output_token_ids_cpu_tensor.numpy()
            self.prompt_token_ids = torch.full((self.expand_max_num_reqs, self.max_penalty_prompt_len),
                                               fill_value = vocab_size,
                                               dtype=torch.int64,
                                               device=self.device)
            self.prompt_token_ids_cpu_tensor = torch.full((self.expand_max_num_reqs, self.max_penalty_prompt_len),
                                               fill_value = vocab_size,
                                               dtype=torch.int64,
                                               device="cpu",
                                               pin_memory=True)
            self.prompt_token_ids_cpu = self.prompt_token_ids_cpu_tensor.numpy()

            self.reorder_batch_threshold = 1 + self.get_spec_k()

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
        else:
            if hasattr(self, "drafter") and isinstance(self.drafter,
                                                    EagleProposer):
                from vllm_gcu.worker.eagle import EagleProposerWithGraph

                self.drafter = EagleProposerWithGraph(self.vllm_config,
                                                    self.device, self,
                                                    self.prepare_next_token_ids_padded_event)

    def get_spec_k(self):
        if not self.vllm_config.additional_config["deepseek_fused_mtp"] \
            or not self.speculative_config:
            return 0
        return self.speculative_config.num_speculative_tokens

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
            self.repetition_penalties_cpu[:expand_reqs] = np.concatenate((repetition_penalties_cpu[:num_decodes].repeat(expand_cnt),
                                                                            repetition_penalties_cpu[num_decodes:]))
            self.repetition_penalties[:expand_reqs].copy_(self.repetition_penalties_cpu_tensor[:expand_reqs], non_blocking=True)

            self.frequency_penalties_cpu[:expand_reqs] = np.concatenate((frequency_penalties_cpu[:num_decodes].repeat(expand_cnt),
                                                                            frequency_penalties_cpu[num_decodes:]))
            self.frequency_penalties[:expand_reqs].copy_(self.frequency_penalties_cpu_tensor[:expand_reqs], non_blocking=True)
            self.presence_penalties_cpu[:expand_reqs] = np.concatenate((presence_penalties_cpu[:num_decodes].repeat(expand_cnt),
                                                                           presence_penalties_cpu[num_decodes:]))
            self.presence_penalties[:expand_reqs].copy_(self.presence_penalties_cpu_tensor[:expand_reqs], non_blocking=True)
            self.output_token_ids_cpu[:expand_reqs,:output_token_ids.shape[1]] = np.concatenate((output_token_ids[:num_decodes].repeat(expand_cnt, axis=0),
                                                                      output_token_ids[num_decodes:]),axis=0)
            self.output_token_ids[:expand_reqs].copy_(self.output_token_ids_cpu_tensor[:expand_reqs], non_blocking=True)
            self.prompt_token_ids_cpu[:expand_reqs,:prompt_token_ids.shape[1]] = np.concatenate((prompt_token_ids[:num_decodes].repeat(expand_cnt, axis=0),
                                                                      prompt_token_ids[num_decodes:]),axis=0)
            self.prompt_token_ids[:expand_reqs].copy_(self.prompt_token_ids_cpu_tensor[:expand_reqs], non_blocking=True)
            


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
            if num_computed_tokens < len(prompt_token_ids):
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
        assert len(self.attn_groups[0]) == 1
        attn_builder = self.attn_groups[0][0].get_metadata_builder()
        attn_builder._num_decodes = num_decodes
        attn_builder._num_prefills = num_prefills
        attn_builder._num_decode_tokens = num_decode_tokens
        attn_builder._num_prefill_tokens = num_prefill_tokens
        attn_builder.reorder_batch_threshold = 1 + spec_k
        return modified_batch

    def calculate_reorder_batch_threshold(self) -> None:
        if self.vllm_config.additional_config["deepseek_fused_mtp"]:
            assert self.vllm_config.speculative_config is not None
            spec_k = self.vllm_config.speculative_config.num_speculative_tokens
            return spec_k + 1
        else:
            return super().calculate_reorder_batch_threshold()

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            # NOTE(lucas): currently no backend supports the custom masking
            #  required for DCP with q_len > 1, so we assert here. Remove this
            #  assert once the custom mask is support is added to FA3.
            if self.dcp_world_size > 1:
                assert self.reorder_batch_threshold == 1, \
                    "DCP not support reorder_batch_threshold > 1 now."
            if self.vllm_config.additional_config["deepseek_fused_mtp"]:
                self.reorder_batch_fused_mtp(self.input_batch,
                                             scheduler_output,
                                             decode_threshold=self.reorder_batch_threshold,
                                             requests = self.requests)
            else:
                reorder_batch_to_split_decodes_and_prefills(
                    self.input_batch,
                    scheduler_output,
                    decode_threshold=self.reorder_batch_threshold)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if self.speculative_config and self.speculative_config.use_eagle() \
            and hasattr(self, "drafter"):
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            if self.device.type == 'xpu':
                get_kv_transfer_group().set_host_xfer_buffer_ops(
                    copy_kv_blocks)

        if self.dcp_world_size > 1:
            layer_names = self.attn_groups[0][0].layer_names
            layers = get_layers_from_vllm_config(self.vllm_config,
                                                 AttentionLayerBase,
                                                 layer_names)
            for layer in layers.values():
                assert layer.impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return"
                    " the softmax lse for decode, but the impl "
                    f"{layer.impl.__class__.__name__} "
                    "does not return the softmax lse for decode.")

    @topstx_wrapper
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
        if self.use_async_scheduling and self.vllm_config.additional_config["deepseek_fused_mtp"]:
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            max_query_len = max(tokens)
            attn_builder = self.attn_groups[0][0].get_metadata_builder()
            assert attn_builder._num_prefills == 0 or max_query_len == self.uniform_decode_query_len
            return self._prepare_inputs_fused_mtp_async(scheduler_output)

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
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

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
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens])

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx +
                                           actual_num_sched].copy_(
                                               req_embeds[start_pos:actual_end]
                                           )

                output_idx += num_sched

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

        num_tokens = [
            self.requests[r].num_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np
        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)

        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            num_draft_tokens = None
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if self.model_config.is_hybrid:
                    num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                        self.input_batch.num_computed_tokens_cpu[req_idx]
                        >= self.input_batch.num_prompt_tokens[req_idx]) else -1)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            if self.model_config.is_hybrid:
                self.num_decode_draft_tokens.np[:
                                                num_reqs] = num_decode_draft_tokens
                self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
                self.num_decode_draft_tokens.copy_to_gpu()

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
        if use_spec_decode and self.model_config.is_hybrid: # only GDN needs this
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

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
                num_logits_indices=logits_indices.size(0),
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
                if use_spec_decode and isinstance(builder,
                                                  GDNAttentionMetadataBuilder):
                    extra_attn_metadata_args = dict(
                        num_accepted_tokens=self.num_accepted_tokens.
                        gpu[:num_reqs],
                        num_decode_draft_tokens_cpu=self.
                        num_decode_draft_tokens.cpu[:num_reqs],
                    )

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

        return (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens, spec_decode_common_attn_metadata,
                max_num_scheduled_tokens, ubatch_slices,
                num_tokens_after_padding)

    def _prepare_inputs_fused_mtp_async(
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
        discard_request_indices = np.array([], dtype = np.int32)
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)
        # save this useless copy
        #self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)

        # Common case (1D positions)
        self.positions.copy_to_gpu(total_num_scheduled_tokens)
        self._adjust_positions(req_indices, total_num_scheduled_tokens, tokens)

        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for req_id, draft_token_ids in (
                scheduler_output.scheduled_spec_decode_tokens.items()):
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)

        #spec_decode_metadata = self._calc_spec_decode_metadata(
        #    num_draft_tokens, cu_num_tokens)
        #logits_indices = spec_decode_metadata.logits_indices
        spec_decode_metadata = None
        logits_indices = None


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

        return (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens, spec_decode_common_attn_metadata,
                max_num_scheduled_tokens, ubatch_slices,
                num_tokens_after_padding)

    def _adjust_positions(
        self,
        req_indices: np.ndarray,
        total_num_scheduled_tokens: int,
        num_scheduled_token_list: List[int]
    ):
        num_reqs = self.input_batch.num_reqs
        # NOTE(guozelin):
        # for async-scheduling with spec-decoding, decoding request's computed_tokens is
        # advanced by rejected but unknown numbers of tokens, so we have to fix positions、
        # seq_lens and slot_mapping since these tensor is calculated base on
        # computed_tokens_cpu.
        # for attention, computed_tokens_cpu mainly affect the prefill part calculation
        # which is not a matter for decoding requests.
        if self.prev_num_rejected_tokens is not None:
            prev_req_id_to_index = self.input_batch.prev_req_id_to_index
            assert prev_req_id_to_index is not None
            prev_common_req_indices = []
            rejected_indices = []
            for req_id, cur_index in self.input_batch.req_id_to_index.items():
                if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                    prev_common_req_indices.append(prev_index)
                    rejected_indices.append(cur_index)
            if len(prev_common_req_indices) > 0:
                num_rejected_tokens = torch.zeros(num_reqs,
                                              dtype=torch.int32,
                                              device = self.device)
                num_scheduled_tokens = torch.tensor(num_scheduled_token_list,
                                              dtype=torch.int32,
                                              pin_memory=self.pin_memory).to(
                                                self.device,
                                                non_blocking=True
                                              )
                rejected_indices_tensor = torch.tensor(rejected_indices,
                                              dtype=torch.int32,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)
                prev_common_req_indices_tensor = \
                    torch.tensor(prev_common_req_indices,
                                dtype=torch.int32,
                                pin_memory=self.pin_memory).to(
                                    self.device,
                                    non_blocking=True
                                )
                num_rejected_tokens.scatter_(
                            dim = 0,
                            index = rejected_indices_tensor,
                            src =
                            self.prev_num_rejected_tokens[
                                prev_common_req_indices_tensor])

                position_delta = torch.repeat_interleave(
                            num_rejected_tokens,
                            num_scheduled_tokens,
                            dim = 0,
                            output_size = total_num_scheduled_tokens)
                self.positions.gpu[:total_num_scheduled_tokens].subtract_(
                                                        position_delta)
                self.seq_lens.gpu[:num_reqs].subtract_(num_rejected_tokens)
                self.input_batch.block_table.compute_slot_mapping_device(
                            req_indices,
                            self.positions.gpu[:total_num_scheduled_tokens])


    @topstx_wrapper
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            reqs_to_add.append(req_state)

        if self.prepare_next_token_ids_padded_event is not None:
            assert not self.vllm_config.additional_config["deepseek_fused_mtp"]
            self.prepare_next_token_ids_padded_event.synchronize()

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            if self.prev_valid_sampled_tokens_count_pinned_cpu[-1] == 1:
                prev_valid_sampled_token_count = self.prev_valid_sampled_tokens_count_pinned_cpu[
                    : self.input_batch.prev_sampled_token_ids.shape[0]
                ].tolist()
                assert self.input_batch.prev_req_id_to_index is not None
                req_idx = self.input_batch.prev_req_id_to_index.get(
                    req_id, None)
                if req_idx is not None and len(req_state.output_token_ids) > 1:
                    prev_draft_tokens_len = len(self.input_batch.prev_sampled_token_ids[req_idx])
                    num_accepted = prev_valid_sampled_token_count[req_idx]
                    num_computed_tokens -= (prev_draft_tokens_len -
                                            num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        self.prev_valid_sampled_tokens_count_pinned_cpu[-1] == 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

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
        if not self.vllm_config.additional_config["deepseek_fused_mtp"]:
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:num_scheduled_tokens],
                scheduler_output.num_scheduled_tokens,
            )
        else:
            prompt_logprobs_dict = {}

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

    def get_dp_padding(self,
                       num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        if self.vllm_config.parallel_config.enable_expert_parallel:
            return 0, None
        else:
            return super().get_dp_padding(num_tokens)

    def initialize_cudagraph_capture(self) -> None:
        super().initialize_cudagraph_capture()
        if self.vllm_config.additional_config["deepseek_fused_mtp"]:
            self.cudagraph_dispatcher.add_cudagraph_key(
                        CUDAGraphMode.FULL,
                        BatchDescriptor(num_tokens=0, uniform_decode=True))

        if hasattr(self, "drafter") and isinstance(self.drafter,
                                                   EagleProposer):
            self.drafter.cudagraph_dispatcher1.initialize_cudagraph_keys(
                self.compilation_config.cudagraph_mode,
                self.uniform_decode_query_len)

            self.drafter.cudagraph_dispatcher2.initialize_cudagraph_keys(
                self.compilation_config.cudagraph_mode, 1)

    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`")
            return 0
        else:
            self.initialize_cudagraph_capture()

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None
            if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()

                compilation_cases = list(reversed(self.cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=False)

            # Capture full cudagraph for uniform decode batches if we
            # don't already have full mixed prefill-decode cudagraphs.
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                cudagraph_mode.separate_routine():
                max_num_tokens = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len
                decode_cudagraph_batch_sizes = [
                    x for x in self.cudagraph_batch_sizes if
                    ((x <= max_num_tokens and x >= self.uniform_decode_query_len) \
                     or (self.vllm_config.additional_config["deepseek_fused_mtp"] and \
                         x == 0))
                ]
                compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True)

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))
        return cuda_graph_size

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
                if self.vllm_config.additional_config["deepseek_fused_mtp"]:
                    if not is_profile:
                        assert num_tokens == num_tokens_after_padding
                    temperature = self.temperature[:num_tokens]
                    top_p = self.top_p[:num_tokens]
                    top_k = self.top_k[:num_tokens]
                    repetition_penalty = self.repetition_penalties[:num_tokens]
                    presence_penalty = self.presence_penalties[:num_tokens]
                    frequency_penalty = self.presence_penalties[:num_tokens]
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
                    )
                    hidden_states = torch.zeros((num_tokens, self.hidden_size), dtype = self.dtype,
                                                 device = self.device)
                    last_hidden_states = hidden_states[-1:, :] if num_tokens > 0 else \
                                            torch.empty((1, self.hidden_size), dtype = self.dtype,
                                                        device = self.device)
                    if not skip_eplb:
                        self.eplb_step(is_dummy=True, is_profile=is_profile)
                    return hidden_states, last_hidden_states
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle() \
                and not self.vllm_config.additional_config["deepseek_fused_mtp"]:
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens,
                                       spec_decode_common_attn_metadata,
                                       cudagraph_runtime_mode)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    @torch.inference_mode()
    @topstx_wrapper
    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
        num_tokens_after_padding: Optional[torch.Tensor] = None,
    ) -> tuple[int, int, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], torch.Tensor,
               Optional[IntermediateTensors], dict[str, Any]]:
        return super()._preprocess(scheduler_output, intermediate_tensors, ubatch_slices, num_tokens_after_padding)

    def _sample(
        self, logits: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata]
    ) -> SamplerOutput:
        self.input_batch.update_async_output_token_ids()
        return super()._sample(logits, spec_decode_metadata)

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

            uniform_decode = (max_query_len
                              == self.uniform_decode_query_len) and (
                                  num_scheduled_tokens
                                  == self.input_batch.num_reqs * max_query_len)
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=uniform_decode)
            cudagraph_runtime_mode, batch_descriptor = \
                self.cudagraph_dispatcher.dispatch(batch_descriptor)

        # This is currently to get around the assert in the DPMetadata
        # where it wants `num_tokens_across_dp` to align with `num_tokens`
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens

        if self.vllm_config.additional_config["deepseek_fused_mtp"]:
            attn_metadata_builder = self.attn_groups[0][0].get_metadata_builder()
            num_decodes = attn_metadata_builder._num_decodes
            num_prefills = attn_metadata_builder._num_prefills
            spec_k = self.get_spec_k()
            batch_size = self.input_batch.num_reqs
            expand_reqs = num_decodes * (spec_k + 1) + num_prefills
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
                ubatch_slices=ubatch_slices,
            ), record_function_or_nullcontext("Forward"), get_tx_ctx("Forward", "green", "VLLM"),
              self.maybe_get_kv_connector_output(scheduler_output) as
              kv_connector_output):
            if self.vllm_config.additional_config["deepseek_fused_mtp"] \
                and get_pp_group().is_last_rank:
                self.prepare_fused_mtp_input(self.input_batch.sampling_metadata,
                                             batch_size,
                                             num_decodes=num_decodes,
                                             num_prefills=num_prefills,
                                             spec_k=spec_k,
                                             scheduled_spec_decode_tokens=scheduler_output.scheduled_spec_decode_tokens,
                                             first_kv_transfer = scheduler_output.first_transfer_request
                                             )

                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    draft_tokens=self.draft_tokens[:batch_size, :spec_k],
                    top_p=self.top_p[:expand_reqs],
                    top_k=self.top_k[:expand_reqs],
                    temperature=self.temperature[:expand_reqs],
                    repetition_penalty=self.repetition_penalties[:expand_reqs],
                    frequency_penalty=self.frequency_penalties[:expand_reqs],
                    presence_penalty=self.presence_penalties[:expand_reqs],
                    prompt_token_ids=self.prompt_token_ids[:expand_reqs],
                    output_token_ids=self.output_token_ids[:expand_reqs],
                )
                if not self.vllm_config.model_config.enforce_eager and \
                    cudagraph_runtime_mode == CUDAGraphMode.FULL and \
                    batch_descriptor.uniform_decode and \
                    (num_input_tokens % (1 + spec_k) == 0) and \
                    num_input_tokens <= self.vllm_config.compilation_config.max_capture_size and num_prefills > 0:
                    assert num_input_tokens // (1 + spec_k) - num_decodes >= num_prefills, \
                        f"some request with less than {1+spec_k} tokens is considered as prefill requests, which is not desirable"
                    # fix prefill accepted_tokens when prefill tokens and decode tokens are mixed when decode cuda graph is used
                    main_model_sampled_ids = model_output["main_model_sampled_tokens"]
                    sampled_token_ids = model_output["accepted_tokens"]
                    accepted_lens = model_output["accepted_lens"]
                    sampled_token_ids = sampled_token_ids[:num_decodes+num_prefills]
                    sampled_token_ids[num_decodes:, :] = torch.full((num_prefills, spec_k+1), fill_value=-1,
                                                                     dtype = sampled_token_ids.dtype,
                                                                     device = sampled_token_ids.device)
                    accepted_lens = accepted_lens[:num_decodes+num_prefills]
                    accepted_lens[num_decodes:] = torch.full((num_prefills,), fill_value = 1,
                                                                 dtype= accepted_lens.dtype,
                                                                 device = accepted_lens.device)
                    index = torch.arange(start = num_decodes, end = num_decodes+num_prefills, dtype = torch.int32, device = self.device)
                    index = torch.index_select(attn_metadata["ds_main_with_mtp"].query_start_loc, dim = 0, index = index)
                    index.sub_(1)
                    sampled_token_ids[num_decodes:, 0] = torch.index_select(main_model_sampled_ids, dim = 0, index = index).squeeze(-1)
                else:
                    sampled_token_ids = model_output["accepted_tokens"]
                    accepted_lens = model_output["accepted_lens"]
                    sampled_token_ids = sampled_token_ids[:num_decodes+num_prefills]
                    accepted_lens = accepted_lens[:num_decodes+num_prefills]
                # avoid modified model_output inplace
                model_output = IntermediateTensors({
                    "main_model_sampled_tokens" : model_output["main_model_sampled_tokens"],
                    "accepted_tokens": sampled_token_ids,
                    "accepted_lens": accepted_lens,
                    "next_draft_tokens": model_output["next_draft_tokens"],
                    "next_token_ids": model_output["next_token_ids"],
                })
            else:
                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

        if self.vllm_config.additional_config["deepseek_fused_mtp"] \
            and get_pp_group().is_last_rank:
            sampled_token_ids = model_output["accepted_tokens"]
            logprobs_tensor = None
            sampler_output = SamplerOutput(sampled_token_ids=sampled_token_ids,
                             logprobs_tensors=logprobs_tensor)
            self._draft_token_ids = model_output["next_draft_tokens"][:num_decodes,:]
            if self.use_async_scheduling:
                if self._draft_token_ids.data_ptr() == self.draft_tokens.data_ptr():
                    self.tmp_draft_token_ids[:num_decodes,:].copy_(self._draft_token_ids, non_blocking = True)
                    self._draft_token_ids = self.tmp_draft_token_ids[:num_decodes,:]
                valid_sampled_tokens_count = model_output["accepted_lens"]
                next_token_ids = model_output["next_token_ids"][:num_decodes+num_prefills]

                self.prev_next_token_ids = next_token_ids.squeeze(1)
                num_draft_tokens = [self.get_spec_k() + 1] * self.input_batch.num_reqs
                if num_prefills > 0:
                    num_draft_tokens[num_decodes:] = [1] * num_prefills
                self.num_rejected_tokens_cpu[:num_decodes+num_prefills] = np.array(num_draft_tokens, dtype = np.int32)
                self.num_rejected_tokens[:num_decodes+num_prefills].copy_(self.num_rejected_tokens_cpu_tensor[:num_decodes+num_prefills],
                                                                          non_blocking = True)
                self.num_rejected_tokens[:num_decodes+num_prefills].subtract_(valid_sampled_tokens_count)
                self.prev_num_rejected_tokens = self.num_rejected_tokens[:num_decodes+num_prefills]
            assert not envs.VLLM_COMPUTE_NANS_IN_LOGITS
            assert not self.input_batch.num_prompt_logprobs
            assert sampler_output.logprobs_tensors is None
            with record_function_or_nullcontext("Bookkeep"), get_tx_ctx("Bookkeep", "green", "VLLM"):
                (
                    num_nans_in_logits,
                    logprobs_lists,
                    valid_sampled_token_ids,
                    prompt_logprobs_dict,
                    req_ids_output_copy,
                    req_id_to_index_output_copy,
                    invalid_req_indices,
                ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                                        None, # logits
                                        None, # hidden_states
                                        num_scheduled_tokens)
        else:
            with record_function_or_nullcontext("Postprocess"), get_tx_ctx("Postprocess", "green", "VLLM"):
                if self.use_aux_hidden_state_outputs:
                    # True when EAGLE 3 is used.
                    hidden_states, aux_hidden_states = model_output
                else:
                    # Common case.
                    hidden_states = model_output
                    aux_hidden_states = None

                if not self.broadcast_pp_output:
                    # Common case.
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

                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)
                else:
                    # Rare case.
                    assert not self.is_pooling_model

                    if not get_pp_group().is_last_rank:
                        all_gather_tensors = {
                            "residual":
                            not is_residual_scattered_for_sp(
                                self.vllm_config, num_input_tokens)
                        }
                        get_pp_group().send_tensor_dict(
                            hidden_states.tensors,
                            all_gather_group=get_tp_group(),
                            all_gather_tensors=all_gather_tensors)
                        logits = None
                    else:
                        sample_hidden_states = hidden_states[logits_indices]
                        logits = self.model.compute_logits(sample_hidden_states)

                    model_output_broadcast_data = {}
                    if logits is not None:
                        model_output_broadcast_data["logits"] = logits.contiguous()

                    model_output_broadcast_data = get_pp_group(
                    ).broadcast_tensor_dict(model_output_broadcast_data,
                                            src=len(get_pp_group().ranks) - 1)
                    assert model_output_broadcast_data is not None
                    logits = model_output_broadcast_data["logits"]

                # Apply structured output bitmasks if present
                if scheduler_output.grammar_bitmask is not None:
                    apply_grammar_bitmask(scheduler_output, self.input_batch,
                                        logits, self.device)

            with record_function_or_nullcontext("Sample"), get_tx_ctx("Sample", "green", "VLLM"):
                sampler_output = self._sample(logits, spec_decode_metadata)


            def propose_draft_token_ids(sampled_token_ids):
                assert spec_decode_common_attn_metadata is not None
                with record_function_or_nullcontext("Draft"), get_tx_ctx("Draft", "green", "VLLM"):
                    self._draft_token_ids = self.propose_draft_token_ids(
                        scheduler_output,
                        sampled_token_ids,
                        self.input_batch.sampling_metadata,
                        hidden_states,
                        sample_hidden_states,
                        aux_hidden_states,
                        spec_decode_metadata,
                        spec_decode_common_attn_metadata,
                    )

            use_padded_batch_for_eagle = self.speculative_config and \
                self.speculative_config.use_eagle() and \
                not self.speculative_config.disable_padded_drafter_batch
            effective_drafter_max_model_len = self.max_model_len
            if effective_drafter_max_model_len is None:
                effective_drafter_max_model_len = self.model_config.max_model_len
            if (self.speculative_config
                    and self.speculative_config.draft_model_config is not None
                    and self.speculative_config.draft_model_config.max_model_len
                    is not None):
                effective_drafter_max_model_len = (
                    self.speculative_config.draft_model_config.max_model_len)
            # pick https://github.com/vllm-project/vllm/pull/25884
            input_fits_in_drafter = spec_decode_common_attn_metadata and (
                spec_decode_common_attn_metadata.max_seq_len +
                self.speculative_config.num_speculative_tokens
                <= effective_drafter_max_model_len)
            if use_padded_batch_for_eagle and input_fits_in_drafter:
                # EAGLE speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                propose_draft_token_ids(sampler_output.sampled_token_ids)


            with record_function_or_nullcontext("Bookkeep"), get_tx_ctx("Bookkeep", "green", "VLLM"):
                (
                    num_nans_in_logits,
                    logprobs_lists,
                    valid_sampled_token_ids,
                    prompt_logprobs_dict,
                    req_ids_output_copy,
                    req_id_to_index_output_copy,
                    invalid_req_indices,
                ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                                        logits, hidden_states,
                                        num_scheduled_tokens)

            if (self.speculative_config and not use_padded_batch_for_eagle
                    and input_fits_in_drafter):
                # ngram and other speculative decoding methods use the sampled
                # tokens on the CPU, so they are run after bookkeeping.
                propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("EPLB"), get_tx_ctx("EPLB", "green", "VLLM"):
            self.eplb_step()

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

        async_output = GCUAsyncGPUModelRunnerOutput(
            vocab_size=self.input_batch.vocab_size,
            event_poll_span_ms= 1 if self.vllm_config.additional_config["deepseek_fused_mtp"] else -1,
            delay_update_output_token_ids=True if self.vllm_config.additional_config["deepseek_fused_mtp"] else False,
            req_ids = self.input_batch.req_ids.copy(),
            requests = self.requests,
            num_output_placeholder = 1 + self.get_spec_k(),
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

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            draft_token_ids = self._draft_token_ids.tolist()
            req_ids = req_ids[:len(draft_token_ids)]
        else:
            draft_token_ids = self._draft_token_ids
        #logger.error(f'draft_token_ids:{draft_token_ids}')
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32)
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32)
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        # [Enflame]: use pin_memory to avoid sync from torch_gcu.transfer_to_gcu
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).pin_memory().to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).pin_memory().to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).pin_memory().to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).pin_memory().to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def _prepare_input_ids(self, total_num_scheduled_tokens: int,
                           cu_num_tokens: np.ndarray) -> None:

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        """add support for spec decoding"""
        if self._draft_token_ids is not None:
            _draft_token_ids = self._draft_token_ids
            if _draft_token_ids.shape[0] == 0:
                # for fused_mtp, prefill requests has no draft_tokens computed
                _draft_token_ids = torch.zeros((self.prev_next_token_ids.shape[0], self.get_spec_k()),
                                                dtype = torch.int32, device = self.device)
            self.input_batch.prev_sampled_token_ids = torch.cat((
                self.prev_next_token_ids.unsqueeze(dim=1),
                _draft_token_ids.to(torch.int32),
            ),
                                                                dim=1)

        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        flattened_indices = []
        prev_common_req_indices = []
        indices_match = True
        max_flattened_index = -1
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                flattened_index = cu_num_tokens[cur_index].item(
                ) - self.uniform_decode_query_len
                flattened_indices.append(flattened_index)
                indices_match &= (
                    prev_index *
                    self.uniform_decode_query_len == flattened_index)
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_common_tokens = len(
            flattened_indices) * self.uniform_decode_query_len
        if num_common_tokens < total_num_scheduled_tokens:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids_cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (
                num_common_tokens - self.uniform_decode_query_len):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.
                prev_sampled_token_ids[:num_common_tokens //
                                       self.uniform_decode_query_len, :self.
                                       uniform_decode_query_len].flatten(),
                non_blocking=True)
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_common_tokens] = True
            return
        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking.
        input_ids_index_tensor = torch.tensor(flattened_indices,
                                              dtype=torch.int64,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)

        for i_s in range(self.uniform_decode_query_len):
            self.input_ids.gpu.scatter_(
                dim=0,
                index=input_ids_index_tensor + i_s,
                src=self.input_batch.prev_sampled_token_ids[
                    prev_common_req_indices_tensor, i_s])

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
         """
        if self.vllm_config.kv_transfer_config is None or \
            self.vllm_config.kv_transfer_config.kv_connector != 'NixlConnector':
            return super()._allocate_kv_cache_tensors(kv_cache_config)

        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.gcu.tops_malloc_host_accessible(
                [kv_cache_tensor.size],
                dtype=torch.int8,
            ).fill_(0)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            layer_names.update(group.layer_names)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def load_model(self, eep_scale_up: bool = False) -> None:
        self.vllm_config.compilation_config.inductor_compile_config[
            "post_grad_custom_post_pass"] = PassManager(self.vllm_config)
        with patch("vllm.compilation.backends.PostGradPassManager",
                   SingletonPostGradPassManager):
            super().load_model(eep_scale_up)
        if get_ep_group().world_size == 1:
            prepare_communication_buffer_for_model_noep(self.model)
        if hasattr(self, "drafter") and hasattr(self.drafter, 'model'):
            prepare_communication_buffer_for_model(self.drafter.model)
            if get_ep_group().world_size == 1:
                prepare_communication_buffer_for_model_noep(self.drafter.model)
