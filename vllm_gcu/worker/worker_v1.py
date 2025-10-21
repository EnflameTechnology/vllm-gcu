import torch
from unittest.mock import patch
import os
import gc
import contextlib
import numpy as np
from typing import Optional, Union, Any
from importlib.util import find_spec

# import vllm.device_allocator
from vllm.utils import MemorySnapshot, GiB_bytes, cdiv
from vllm.model_executor import set_random_seed
from vllm.distributed.parallel_state import get_pp_group, get_ep_group, prepare_communication_buffer_for_model
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm.v1.utils import report_usage_stats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.sequence import IntermediateTensors

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.forward_context import BatchDescriptor
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.ubatch_splitting import (check_ubatch_thresholds,
                                             ubatch_split)
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport, AttentionMetadataBuilder, CommonAttentionMetadata,
    create_fast_prefill_custom_backend,
    reorder_batch_to_split_decodes_and_prefills, split_attn_metadata)
from vllm_gcu.utils import (set_gcu_forward_context,
                            dump_memory_snapshot_when_exception,
                            prepare_communication_buffer_for_model_noep,)
from vllm_gcu.compilation.pass_manager import PassManager, SingletonPostGradPassManager
import vllm_gcu.envs as gcu_envs

with patch("vllm.forward_context.set_forward_context", set_gcu_forward_context):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment


class GCUModelRunner(GPUModelRunner):
    def get_dp_padding(self,
                       num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        if self.vllm_config.parallel_config.enable_expert_parallel:
            return 0, None
        else:
            return super().get_dp_padding(num_tokens)

    @torch.inference_mode()
    @dump_memory_snapshot_when_exception('step')
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
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
        assert cudagraph_runtime_mode in {
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
        else:
            num_tokens_across_dp = num_tokens_after_padding
            num_tokens_after_padding = int(num_tokens_after_padding[0].item())


        attn_metadata: Optional[dict[str, Any]] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
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
                num_scheduled_tokens)
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
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id].get_device_tensor(num_reqs),
                    slot_mapping=self.input_batch.
                    block_table[kv_cache_group_id].slot_mapping.gpu[:num_tokens],
                    causal=True)

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

            with self.maybe_randomize_inputs(input_ids), set_gcu_forward_context(
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
                self.drafter.dummy_run(num_tokens)

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
    @dump_memory_snapshot_when_exception('step')
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        return super().execute_model(scheduler_output, intermediate_tensors)

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
        self.vllm_config.compilation_config.inductor_compile_config["post_grad_custom_post_pass"] = PassManager(self.vllm_config)
        with patch("vllm.compilation.backends.PostGradPassManager", SingletonPostGradPassManager):
            super().load_model(eep_scale_up)
        if get_ep_group().world_size == 1:
            prepare_communication_buffer_for_model_noep(self.model)
        if hasattr(self, "drafter") and hasattr(self.drafter, 'model'):
            prepare_communication_buffer_for_model(self.drafter.model)
            if get_ep_group().world_size == 1:
                prepare_communication_buffer_for_model_noep(self.drafter.model)


class GCUWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        import vllm_gcu.kernels  # noqa: F401
        import vllm_gcu.compilation  # noqa: F401
        import vllm_gcu.patch  # noqa: F401
        import vllm_gcu.distributed  # noqa
        if gcu_envs.VLLM_GCU_RANK_LOG_PATH:
            # before init dist, since we want to split eccl init logs
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            world_size = vllm_config.parallel_config.world_size
            rank_across_dp = dp_rank * world_size + rank
            f = open(os.path.join(gcu_envs.VLLM_GCU_RANK_LOG_PATH,
                                  f'worker_{rank_across_dp}.log'),
                     'w', buffering=1)
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
        if vllm_config.additional_config.get('set_cpu_affinity', False):
            current_platform.set_cpu_affinity(local_rank)

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

    def init_device(self):
        os.environ["TORCH_ECCL_AVOID_RECORD_STREAMS"] = "1"

        self.device = torch.device(f"gcu:{self.local_rank}")
        current_platform.set_device(self.device)

        current_platform.check_if_supports_dtype(self.model_config.dtype)
        gc.collect()
        torch.gcu.empty_cache()

        self.init_snapshot = MemorySnapshot()
        self.requested_memory = (
            self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        )

        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)  # noqa: E731
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )

        init_worker_distributed_environment(
            self.vllm_config, self.rank, self.distributed_init_method, self.local_rank,
            current_platform.dist_backend,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: GCUModelRunner = GCUModelRunner(
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output,
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        has_tx = find_spec("topstx") is not None
        if has_tx:
            import topstx
            tx_ctx = topstx.annotate(f"execute_{num_scheduled_tokens}", color="green", domain="VLLM")
        else:
            tx_ctx = contextlib.nullcontext()

        with tx_ctx:
            return super().execute_model(scheduler_output)


    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(0)
