"""A GCU worker class."""

import os
import gc
from typing import Optional, Tuple, Type
from datetime import timedelta

import torch
import torch.distributed
import torch_gcu  # noqa: F401

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_kv_transfer_initialized,
    ensure_model_parallel_initialized,
    get_dp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.utils import GiB_bytes, memory_profiling, MemorySnapshot
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.worker.worker import Worker

from vllm_gcu.worker.model_runner import GCUModelRunner
import vllm_gcu.envs as gcu_envs


logger = init_logger(__name__)


class GCUWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        import vllm_gcu.kernels  # noqa: F401
        import vllm_gcu.patch.worker  # noqa: F401
        import vllm_gcu.compilation  # noqa: F40

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

        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
            model_runner_cls,
        )
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = (
            {}
            if speculative_config is None
            or (
                speculative_config.draft_model_config.hf_config.model_type
                == model_config.hf_config.model_type
            )
            or (
                speculative_config.draft_model_config.hf_config.model_type
                not in ("medusa", "mlp_speculator", "eagle", "deepseek_mtp")
            )
            else {"return_hidden_states": True}
        )

        if (
            model_config.runner_type != "pooling"
            and not model_config.is_encoder_decoder
        ):
            self.model_runner = GCUModelRunner(
                vllm_config=self.vllm_config,
                kv_cache_dtype=self.cache_config.cache_dtype,
                is_driver_worker=is_driver_worker,
                **speculative_args,
            )

    def init_device(self) -> None:

        self.device = torch.device(f"gcu:{self.local_rank}")
        torch.gcu.set_device(self.device)
        gc.collect()
        torch.gcu.empty_cache()
        torch.gcu.reset_peak_memory_stats()
        self.baseline_snapshot = MemorySnapshot()

        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        torch.gcu.empty_cache()
        torch.gcu.reset_peak_memory_stats()
        if self.cache_config.num_gpu_blocks_override:
            cache_block_size = self.get_cache_block_size_bytes()
            num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
            return self.cache_config.num_gpu_blocks_override, num_cpu_blocks

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
        if (
            self.parallel_config.enable_expert_parallel
            and self.parallel_config.data_parallel_size > 1
        ):
            blocks_tensor = torch.tensor([num_gpu_blocks, num_cpu_blocks], dtype=torch.int32)
            torch.distributed.all_reduce(
                blocks_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_dp_group().cpu_group,
            )
            num_gpu_blocks = blocks_tensor[0].item()
            num_cpu_blocks = blocks_tensor[1].item()

        return num_gpu_blocks, num_cpu_blocks


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""

    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        distributed_init_method,
        local_rank,
        backend="eccl",
    )

    # ugly WA as bug in 0.8.0
    parallel_config.world_size = parallel_config.world_size_across_dp
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
    )
    parallel_config.world_size = (
        parallel_config.pipeline_parallel_size * parallel_config.tensor_parallel_size
    )

    ensure_kv_transfer_initialized(vllm_config)

    group = get_tp_group()
    group.use_custom_op_call = False

    group = get_pp_group()
    group.use_custom_op_call = False

    group = get_dp_group()
    group.use_custom_op_call = False

    all_ranks = torch.arange(parallel_config.world_size_across_dp).reshape(parallel_config.data_parallel_size, -1)
    group_ranks = all_ranks.transpose(0, 1).unbind(0)
    rank_cpu_group = None
    for ranks in group_ranks:
        # timedelta.max overflow chrono::milliseconds
        cpu_group = torch.distributed.new_group(ranks.tolist(),
                                                backend="gloo",
                                                timeout=timedelta(days=100*365)
                                                )
        if group.rank in ranks:
            rank_cpu_group = cpu_group
    group.cpu_group = rank_cpu_group

    group = get_world_group()
    group.use_custom_op_call = False
