import gc
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Union, Any, cast, List
import time
from unittest.mock import patch
import numpy as np
import torch
from vllm.utils import (cdiv, length_from_prompt_token_ids_or_embeds)
from vllm.v1.attention.backends.utils import AttentionCGSupport
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

    def __init__(self, vocab_size: int, event_poll_span_ms = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.event_poll_span_ms = event_poll_span_ms

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
        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output

class GCUModelRunner(GPUModelRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.positions_tensor: torch.Tensor | None = None

        logprobs_mode = self.sampler.topk_topp_sampler.logprobs_mode
        self.sampler = GCUSampler(logprobs_mode)
        
        if hasattr(self, "rejection_sampler"):
            self.rejection_sampler = GCURejectionSampler()
        
        if hasattr(self, "drafter") and isinstance(self.drafter,
                                                EagleProposer):
            from vllm_gcu.worker.eagle import EagleProposerWithGraph

            self.drafter = EagleProposerWithGraph(self.vllm_config,
                                                    self.device, self)
        
        self.uses_xdrope_dim = self.model_config.uses_xdrope_dim
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

    def get_spec_k(self):
        if not self.speculative_config:
            return 0
        return self.speculative_config.num_speculative_tokens

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
        ret = super()._prepare_inputs(scheduler_output)
        if self.positions_tensor is not None:
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            self.positions.gpu[:total_num_scheduled_tokens].copy_(self.positions_tensor)
            self.positions_tensor = None
        return ret

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        super()._update_states(scheduler_output)
        
        if self.uses_xdrope_dim > 0:
            for new_req_data in scheduler_output.scheduled_new_reqs:
                req_id = new_req_data.req_id
                req_state = self.requests[req_id]
                self._init_xdrope_positions(req_state)

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        # Check if model supports XD-RoPE by verifying method existence
        assert hasattr(model, 'get_xdrope_input_positions'), \
            "XD-RoPE support is not implemented."

        req_state.xdrope_positions = model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

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
            num_draft_tokens = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                if i not in invalid_req_indices_set:
                    num_draft_tokens.append(
                        1 + len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                    )
                else:
                    num_draft_tokens.append(0)

            self.input_batch.prev_num_sampled_tokens = torch.tensor(
                num_draft_tokens,
                dtype=torch.int32,
                pin_memory=self.pin_memory).to(self.device, non_blocking=True)

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
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]
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
            req_state.output_token_ids.extend(sampled_ids)

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
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL \
            and cudagraph_mode.separate_routine() and 0 in self.cudagraph_batch_sizes:
            self.cudagraph_dispatcher.add_cudagraph_key(
                CUDAGraphMode.FULL,
                BatchDescriptor(num_tokens=0, uniform_decode=True))

        min_cg_support = AttentionCGSupport.ALWAYS
        for attn_group in self._attn_group_iterator():
            builder = attn_group.get_metadata_builder()
            if builder.cudagraph_support.value < min_cg_support.value:
                min_cg_support = builder.cudagraph_support
        if min_cg_support == AttentionCGSupport.UNIFORM_BATCH:
            self.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.FULL] = {
                x for x in self.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
                if x.num_tokens % self.uniform_decode_query_len == 0
            }

        if hasattr(self, "drafter") and isinstance(self.drafter,
                                                   EagleProposer):
            self.drafter.cudagraph_dispatcher1.initialize_cudagraph_keys(
                self.compilation_config.cudagraph_mode,
                self.uniform_decode_query_len)

            self.drafter.cudagraph_dispatcher2.initialize_cudagraph_keys(
                self.compilation_config.cudagraph_mode, 1)
            if min_cg_support == AttentionCGSupport.UNIFORM_BATCH:
                supported = {i.num_tokens for i in self.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.FULL]}
                self.drafter.cudagraph_dispatcher1.cudagraph_keys[CUDAGraphMode.FULL] = {
                    x for x in self.drafter.cudagraph_dispatcher1.cudagraph_keys[CUDAGraphMode.FULL]
                    if x.num_tokens in supported
                }
                self.drafter.cudagraph_dispatcher2.cudagraph_keys[CUDAGraphMode.FULL] = {
                    x for x in self.drafter.cudagraph_dispatcher2.cudagraph_keys[CUDAGraphMode.FULL]
                    if x.num_tokens * self.uniform_decode_query_len in supported
                }

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
        compiled_cases = []
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None
            if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()

                compilation_cases = list(reversed(self.cudagraph_batch_sizes))
                compiled_cases += compilation_cases
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
                cudagraph_keys = {i.num_tokens for i in self.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.FULL]}
                decode_cudagraph_batch_sizes = [
                    x for x in self.cudagraph_batch_sizes if
                    ((x <= max_num_tokens and x >= self.uniform_decode_query_len) \
                     or x == 0) and x in cudagraph_keys
                ]
                compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                compiled_cases += compilation_cases_decode
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

        # NOTE: some cases need be compiled may be filtered out by cudagraph conditions
        compilation_cases = set(
            self.vllm_config.compilation_config.compile_sizes) & (
                set(self.cudagraph_batch_sizes) - set(compiled_cases))
        compilation_cases = sorted(compilation_cases, reverse=True)
        for num_tokens in compilation_cases:
            for _ in range(self.compilation_config.cudagraph_num_of_warmups + 1):
                self._dummy_run(num_tokens,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                uniform_decode=False,
                                allow_microbatching=False,
                                skip_eplb=True,
                                remove_lora=False)

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
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
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
                            # set back max_seq_len=max_model_len for attn backends need it,
                            # for cudagraph compatibility
                            common_attn_metadata.max_seq_len = self.max_model_len
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
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_after_padding]
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

            if self.speculative_config and self.speculative_config.use_eagle():
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
        
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if ubatch_slices:
            assert num_tokens_after_padding is not None
            num_input_tokens = int(num_tokens_after_padding[0].item() * 2)
            self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
        elif ubatch_slices is None:
            num_input_tokens = self._get_num_input_tokens(num_scheduled_tokens)
            num_pad, num_tokens_after_padding = self.get_dp_padding(
                num_input_tokens)
            num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if (self.supports_mm_inputs and get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder):
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)

            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:num_scheduled_tokens] \
                .nonzero(as_tuple=False) \
                .squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.get_input_embeddings(
                    input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs(num_input_tokens)
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        if (self.model_config.is_encoder_decoder
                and scheduler_output.scheduled_encoder_inputs):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_after_padding,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        )

    def _sample(
        self, logits: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata]
    ) -> SamplerOutput:
        self.input_batch.update_async_output_token_ids()
        return super()._sample(logits, spec_decode_metadata)

    def _model_forward(self, input_ids, input_positions, intermediate_tensors, inputs_embeds, model_kwargs, extra_args):
        model_output = self.model(
            input_ids=input_ids,
            positions=input_positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        if self.use_aux_hidden_state_outputs:
            # True when EAGLE 3 is used.
            hidden_states, aux_hidden_states = model_output
        else:
            # Common case.
            hidden_states = model_output
            aux_hidden_states = None

        return hidden_states, aux_hidden_states

    def _determine_batch_descriptor(self, extra_args):
        max_query_len = extra_args['max_query_len']
        num_scheduled_tokens = extra_args['num_scheduled_tokens']
        num_input_tokens = extra_args['num_input_tokens']

        uniform_decode = (max_query_len
                            == self.uniform_decode_query_len) and (
                                num_scheduled_tokens
                                == self.input_batch.num_reqs * max_query_len)
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                            uniform_decode=uniform_decode)
        return batch_descriptor

    def _compute_logist(self, hidden_states, extra_args):
        num_scheduled_tokens = extra_args['num_scheduled_tokens']
        num_scheduled_tokens_np = extra_args['num_scheduled_tokens_np']
        logits_indices = extra_args['logits_indices']
        num_input_tokens = extra_args['num_input_tokens']

        with record_function_or_nullcontext("Postprocess"), get_tx_ctx("Postprocess", "green", "VLLM", "execute"):
            if not self.broadcast_pp_output:
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
            return logits

    def _compute_sampler_output(self, logist, spec_decode_metadata, extra_args):
        sampler_output = self._sample(logist, spec_decode_metadata)
        return sampler_output

    def _compute_spect_tokens(self, hidden_states, aux_hidden_states, sampler_output, extra_args):
        spec_decode_common_attn_metadata = extra_args['spec_decode_common_attn_metadata']
        spec_decode_metadata = extra_args['spec_decode_metadata']
        valid_sampled_token_ids = extra_args['valid_sampled_token_ids']
        scheduler_output = extra_args['scheduler_output']
        logits_indices = extra_args['logits_indices']

        sample_hidden_states = hidden_states[logits_indices]
        
        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"), get_tx_ctx("Draft", "green", "VLLM", "execute"):
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
        elif (self.speculative_config and not use_padded_batch_for_eagle
                and input_fits_in_drafter):
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

    @torch.inference_mode()
    @dump_memory_snapshot_when_exception('step')
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"), get_tx_ctx("Preprocess", "green", "VLLM", "execute"):
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
            ), record_function_or_nullcontext("Forward"), get_tx_ctx("Forward", "green", "VLLM", "execute"),
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

        with record_function_or_nullcontext("Bookkeep"), get_tx_ctx("Bookkeep", "green", "VLLM", "execute"):
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

        async_output = GCUAsyncGPUModelRunnerOutput(
            vocab_size=self.input_batch.vocab_size,
            event_poll_span_ms=1,
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream
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

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's xdrope_positions are pre-computed
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's xdrope_positions on-the-fly
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len
                context_len=num_computed_tokens + prompt_part_len
                num_new_tokens=completion_part_len

                values = np.arange(
                    context_len,
                    context_len + num_new_tokens,
                    dtype=self.xdrope_positions.np.dtype,
                )
                self.xdrope_positions.np[:, dst_start : dst_start + num_new_tokens] = values

                xdrope_pos_ptr += completion_part_len
    
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

        # calc actual positions/seq_lens/block_table/slot_mapping before input_ids
        if not self.uses_mrope:
            self.positions.copy_to_gpu(total_num_scheduled_tokens)
        self.positions_tensor = self.positions.gpu[:total_num_scheduled_tokens].clone().to(torch.int32)

        num_scheduled_tokens = np.diff(cu_num_tokens, prepend=0)
        num_scheduled_tokens_tensor = torch.tensor(
            num_scheduled_tokens.tolist(),
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        req_indices = np.repeat(self.arange_np[:len(cu_num_tokens)], num_scheduled_tokens)
        req_indices_tensor = torch.tensor(
            req_indices.tolist(),
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)

        prev_num_reqs, prev_max_gen_len = self.input_batch.prev_sampled_token_ids.shape
        prev_num_rejected_tokens = torch.zeros([prev_num_reqs], dtype=torch.int32, device=self.device)
        if self.input_batch.prev_valid_sampled_tokens_count is not None:
            prev_num_rejected_tokens = (self.input_batch.prev_valid_sampled_tokens_count - self.input_batch.prev_num_sampled_tokens).to(torch.int32)

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        """add support for spec decoding"""
        if self._draft_token_ids is not None:
            _draft_token_ids = self._draft_token_ids
            self.input_batch.prev_sampled_token_ids = torch.cat((
                self.input_batch.prev_next_token_ids.unsqueeze(dim=1),
                _draft_token_ids.to(torch.int32),
            ),
                                                                dim=1)

        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None

        flattened_indices = []
        flattened_prev_indices = []

        curr_common_req_indices = []
        prev_common_req_indices = []

        common_num_scheduled_tokens = []

        indices_match = True
        max_flattened_index = -1
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                curr_common_req_indices.append(cur_index)

                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                flattened_index_start = cu_num_tokens[cur_index-1].item() if cur_index > 0 else 0
                req_tokens = num_scheduled_tokens[cur_index]
                common_num_scheduled_tokens.append(req_tokens)
                for i in range(req_tokens):
                    flattened_prev_indices.append(prev_index * prev_max_gen_len + i)
                    flattened_indices.append(flattened_index_start+i)
                indices_match &= (
                    prev_index *
                    self.uniform_decode_query_len == flattened_index_start)
                indices_match &= (req_tokens == self.uniform_decode_query_len)
                max_flattened_index = max(max_flattened_index, flattened_index_start+req_tokens-1)
        num_common_tokens = len(flattened_indices)

        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking.
        input_ids_index_tensor = torch.tensor(flattened_indices,
                                              dtype=torch.int64,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)
        prev_sampled_ids_index_tensor = torch.tensor(flattened_prev_indices,
                                              dtype=torch.int64,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)

        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        cur_common_req_indices_tensor = torch.tensor(
            curr_common_req_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        common_num_scheduled_tokens_tensor = torch.tensor(
            common_num_scheduled_tokens,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)

        flattened_num_rejected_tokens = torch.repeat_interleave(
            prev_num_rejected_tokens[prev_common_req_indices_tensor],
            common_num_scheduled_tokens_tensor,
            dim=0,
            output_size=num_common_tokens)

        self.positions_tensor.scatter_add_(
            dim=0,
            index=input_ids_index_tensor,
            src=flattened_num_rejected_tokens,
        )
        self.seq_lens.gpu.scatter_add_(
            dim=0,
            index=cur_common_req_indices_tensor,
            src=prev_num_rejected_tokens[prev_common_req_indices_tensor],
        )
        self.input_batch.block_table.compute_slot_mapping(req_indices_tensor, self.positions_tensor)

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
        if indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            num_common_reqs = num_common_tokens // self.uniform_decode_query_len
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.
                prev_sampled_token_ids[:num_common_reqs, :self.uniform_decode_query_len].flatten(),
                non_blocking=True)
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_common_tokens] = True
            return

        self.input_ids.gpu.scatter_(
            dim=0,
            index=input_ids_index_tensor,
            src=self.input_batch.prev_sampled_token_ids.flatten()[
                prev_sampled_ids_index_tensor])

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
