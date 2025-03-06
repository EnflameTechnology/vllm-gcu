"""A GPU worker class."""

import gc
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed
import torch_gcu

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    ensure_kv_transfer_initialized,
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (
    ExecuteModelRequest,
    IntermediateTensors,
    SequenceGroupMetadata,
    SequenceGroupMetadataDelta,
)
from vllm.utils import bind_kv_cache, GiB_bytes
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner

from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.worker.pooling_model_runner import PoolingModelRunner
from vllm.worker.worker_base import (
    LocalOrDistributedWorkerBase,
    WorkerBase,
    WorkerInput,
)

from vllm_gcu.distributed import PyEcclCommunicator

from vllm_gcu.utils import memory_profiling, MemorySnapshot

from vllm_gcu.worker.model_runner import GCUModelRunner


logger = init_logger(__name__)


class GCUWorker(LocalOrDistributedWorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[ModelRunnerBase]] = None,
    ) -> None:
        import vllm_gcu.kernels

        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if is_driver_worker:
            assert (
                rank % self.parallel_config.tensor_parallel_size == 0
            ), "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = (
            {}
            if speculative_config is None
            or (speculative_config.draft_model_config.model == model_config.model)
            or (
                speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]
            )
            else {"return_hidden_states": True}
        )

        ModelRunnerClass = GCUModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = PoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True
                ),
            )
        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def init_device(self) -> None:
        import vllm_gcu.envs as gcu_envs

        data_parallel_size = gcu_envs.VLLM_GCU_DATA_PARALLEL_SIZE
        data_parallel_rank = gcu_envs.VLLM_GCU_DATA_PARALLEL_RANK

        global_world_size = self.parallel_config.world_size * data_parallel_size
        unique_rank = self.parallel_config.world_size * data_parallel_rank + self.rank
        device_rank = unique_rank % torch.gcu.device_count()

        self.device = torch.device(f"gcu:{device_rank}")
        torch.gcu.set_device(self.device)
        gc.collect()
        torch.gcu.empty_cache()
        torch.gcu.reset_peak_memory_stats()
        self.baseline_snapshot = MemorySnapshot()

        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            global_world_size,
            unique_rank,
            self.distributed_init_method,
            self.local_rank,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        torch.gcu.empty_cache()
        torch.gcu.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.gcu.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.baseline_snapshot, weights_memory=self.model_runner.model_memory_usage
        ) as result:
            self.model_runner.profile_run()

        memory_for_current_instance = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization
        )
        available_kv_cache_memory = (
            memory_for_current_instance - result.non_kv_cache_memory
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        msg = (
            f"Memory profiling takes {result.profile_time:.2f} seconds\n"
            "the current vLLM instance can use "
            "total_gpu_memory "
            f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
            " x gpu_memory_utilization "
            f"({self.cache_config.gpu_memory_utilization:.2f})"
            f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
            "model weights take "
            f"{(result.weights_memory / GiB_bytes):.2f}GiB;"
            " non_torch_memory takes "
            f"{(result.non_torch_increase / GiB_bytes):.2f}GiB;"
            " PyTorch activation peak memory takes "
            f"{(result.torch_peak_increase / GiB_bytes):.2f}GiB;"
            " the rest of the memory reserved for KV Cache is "
            f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB."
        )

        logger.info(msg)
        # Final cleanup
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(
            num_gpu_blocks,
            self.cache_config.block_size,
            self.cache_config.is_attention_free,
            self.model_config.max_model_len,
        )

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(
                self.cache_config,
                self.model_config,
                self.parallel_config,
                self.device_config,
            )
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
        bind_kv_cache(self.compilation_config.static_forward_context, self.gpu_cache)

    def _warm_up_model(self) -> None:
        # warm up sizes that are not in topsgraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x
                for x in warmup_sizes
                if x not in self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
        self, execute_model_req: ExecuteModelRequest
    ) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(
            execute_model_req.blocks_to_swap_in, device="cpu", dtype=torch.int64
        ).view(-1, 2)
        blocks_to_swap_out = torch.tensor(
            execute_model_req.blocks_to_swap_out, device="cpu", dtype=torch.int64
        ).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(
            execute_model_req.blocks_to_copy, device=self.device, dtype=torch.int64
        ).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (
            worker_input.blocks_to_swap_in is not None
            and worker_input.blocks_to_swap_in.numel() > 0
        ):
            self.cache_engine[virtual_engine].swap_in(worker_input.blocks_to_swap_in)
        if (
            worker_input.blocks_to_swap_out is not None
            and worker_input.blocks_to_swap_out.numel() > 0
        ):
            self.cache_engine[virtual_engine].swap_out(worker_input.blocks_to_swap_out)
        if (
            worker_input.blocks_to_copy is not None
            and worker_input.blocks_to_copy.numel() > 0
        ):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    def _get_cached_seq_group_metadata(
        self,
        seq_group_metadata_list: List[
            Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]
        ],
        finished_request_ids: List[str],
    ) -> List[SequenceGroupMetadata]:
        """Return a list of cached Sequence Group Metadata after updating its
        state.

        It is used because scheduler only sends delta to workers to reduce
        the data payload size. The function also cleans up cache based on
        a given `finished_request_ids`.
        """
        new_seq_group_metadata_list = []
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id not in self._seq_group_metadata_cache:
                # The first prefill.
                assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                self._seq_group_metadata_cache[request_id] = metadata_or_delta
            else:
                # The first prefill is already cached.
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    self._seq_group_metadata_cache[request_id].apply_delta(
                        metadata_or_delta
                    )
                else:
                    # If metadata snapshot is sent again, it is
                    # preempted. Reset the cache because we need to start
                    # from scratch.
                    assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                    self._seq_group_metadata_cache[request_id] = metadata_or_delta

            new_seq_group_metadata_list.append(
                self._seq_group_metadata_cache[request_id]
            )

        # Clean up finished ids
        for finished_id in finished_request_ids:
            del self._seq_group_metadata_cache[finished_id]

        return new_seq_group_metadata_list

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:
        if execute_model_req is not None:
            new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.finished_requests_ids,
            )

            execute_model_req.seq_group_metadata_list = new_seq_group_metadata_list
        output = super()._execute_model_spmd(execute_model_req, intermediate_tensors)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_runner.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.remove_lora(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.model_runner.list_prompt_adapters()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes."""
        return CacheEngine.get_cache_block_size(
            self.cache_config, self.model_config, self.parallel_config
        )


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    world_size: int,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    from vllm.utils import get_distributed_init_method

    import vllm_gcu.envs as gcu_envs
    from vllm_gcu.distributed.parallel_state import initialize_data_parallel

    parallel_config = vllm_config.parallel_config

    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    data_parallel_size = gcu_envs.VLLM_GCU_DATA_PARALLEL_SIZE
    data_parallel_rank = gcu_envs.VLLM_GCU_DATA_PARALLEL_RANK

    if data_parallel_size > 1:
        # when DP, use VLLM_GCU_PORT to establish connection
        distributed_init_method = get_distributed_init_method(
            gcu_envs.VLLM_GCU_HOST_ID, gcu_envs.VLLM_GCU_PORT
        )

    init_distributed_environment(
        world_size,
        rank,
        distributed_init_method,
        rank,
        backend="eccl",
    )
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size * data_parallel_size,
    )
    ensure_kv_transfer_initialized(vllm_config)

    group = get_tp_group()
    group.device = torch.device(f"gcu:{group.local_rank % torch.gcu.device_count()}")
    group.pynccl_comm = PyEcclCommunicator(group=group.cpu_group, device=group.device)

    group = get_pp_group()
    group.device = torch.device(f"gcu:{group.local_rank % torch.gcu.device_count()}")
    group.pynccl_comm = PyEcclCommunicator(group=group.cpu_group, device=group.device)
    if data_parallel_size > 1:
        # disable pp when dp
        ranks = [group.local_rank]
        group.device_group = torch.distributed.new_group(ranks, backend="eccl")
        group.cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        group.ranks = ranks
        group.world_size = len(ranks)
        group.rank_in_group = ranks.index(group.local_rank)

    group = get_world_group()
    group.device = torch.device(f"gcu:{group.local_rank % torch.gcu.device_count()}")
    group.pynccl_comm = PyEcclCommunicator(group=group.cpu_group, device=group.device)


def raise_if_cache_size_invalid(
    num_gpu_blocks, block_size, is_attention_free, max_model_len
) -> None:
    if is_attention_free and num_gpu_blocks != 0:
        raise ValueError(
            "No memory should be allocated for the cache blocks "
            f"for an attention-free model, but {num_gpu_blocks}"
            "blocks are allocated."
        )
    if not is_attention_free and num_gpu_blocks <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when "
            "initializing the engine."
        )
    max_seq_len = block_size * num_gpu_blocks
    if not is_attention_free and max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine."
        )
