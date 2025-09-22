import torch
from unittest.mock import patch
import os
import gc
import contextlib
import numpy as np
from typing import Optional, Union
from importlib.util import find_spec

# import vllm.device_allocator
from vllm.utils import MemorySnapshot, GiB_bytes
from vllm.model_executor import set_random_seed
from vllm.distributed.parallel_state import get_pp_group, get_ep_group
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm.v1.utils import report_usage_stats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.sequence import IntermediateTensors

from vllm_gcu import gcumem
from vllm_gcu.utils import (set_gcu_forward_context,
                            dump_memory_snapshot_when_exception,
                            prepare_communication_buffer_for_model_noep,)
import vllm_gcu.envs as gcu_envs

with patch("vllm.forward_context.set_forward_context", set_gcu_forward_context):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


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
        capture_attn_cudagraph: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata
        from vllm.v1.spec_decode.eagle import EagleProposer

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad
        if num_tokens > 0:

            # Set num_scheduled_tokens based on num_tokens and max_num_seqs
            # for dummy run with LoRA so that the num_reqs collectively
            # has num_tokens in total.
            assert num_tokens <= self.scheduler_config.max_num_batched_tokens
            max_num_reqs = self.scheduler_config.max_num_seqs
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
            assert sum(num_scheduled_tokens_list) == num_tokens
            assert len(num_scheduled_tokens_list) == num_reqs
            num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        else:
            num_reqs = 1
            num_scheduled_tokens = np.array([], dtype=np.int32)

        attn_metadata = None
        if capture_attn_cudagraph:
            attn_metadata = {}

            query_start_loc = self.query_start_loc[: num_reqs + 1]
            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(
                self.seq_lens_cpu[:num_reqs], non_blocking=True
            )
            seq_lens = self.seq_lens[:num_reqs]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=num_tokens,
            )

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):
                attn_metadata_i = self.attn_metadata_builders[
                    kv_cache_group_id
                ].build_for_cudagraph_capture(common_attn_metadata)
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False
                )

            with (
                self.maybe_randomize_inputs(input_ids),
                set_gcu_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    is_dummy=True,
                ),
            ):
                outputs = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
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

    def load_model(self) -> None:
        super().load_model()
        if get_ep_group().world_size == 1:
            prepare_communication_buffer_for_model_noep(self.model)


with patch("vllm.device_allocator", "cumem", gcumem):
    from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment


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

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

    def init_device(self):
        os.environ["TORCH_ECCL_AVOID_RECORD_STREAMS"] = "1"

        self.device = torch.device(f"gcu:{self.local_rank}")
        torch.gcu.set_device(self.device)
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
            self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
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
