#!/usr/bin/env python
# coding=utf-8

import gc

import torch
import torch_gcu  # noqa: F401
import vllm.envs as envs

from vllm.config import set_current_vllm_config, VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.worker_base import WorkerBase
from vllm_gcu.v1.worker.gcu_model_runner import GCUModelRunner
from vllm_gcu.worker.worker import init_worker_distributed_environment

logger = init_logger(__name__)


class Worker(WorkerBase):
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
        import vllm_gcu.patch.worker  # noqa: F401

        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            raise NotImplementedError
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    def wake_up(self) -> None:
        raise NotImplementedError

    def init_device(self):
        self.device = torch.device(f"gcu:{self.local_rank}")
        torch.gcu.set_device(self.device)
        gc.collect()
        torch.gcu.empty_cache()
        torch.gcu.reset_peak_memory_stats()

        self.init_gpu_memory = torch.gcu.mem_get_info()[0]

        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

        self.model_runner = GCUModelRunner(self.vllm_config, self.device)

    def load_model(self) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        torch.gcu.empty_cache()
        torch.gcu.reset_peak_memory_stats()

        _, total_gpu_memory = torch.gcu.mem_get_info()
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        free_gpu_memory, _ = torch.gcu.mem_get_info()
        assert self.init_gpu_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GCU memory was "
            "not properly cleaned up before initializing the vLLM instance."
        )

        peak_memory = torch.gcu.memory_stats()["allocated_bytes.all.peak"]

        torch.gcu.empty_cache()
        torch_allocated_bytes = torch.gcu.memory_stats()["allocated_bytes.all.current"]
        total_allocated_bytes = (
            torch.gcu.mem_get_info()[1] - torch.gcu.mem_get_info()[0]
        )
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory
        )

        return int(available_kv_cache_memory)

    def get_kv_cache_spec(self):
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        from contextlib import nullcontext

        context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
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
            self.model_runner.capture_model()

        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )
            self.model_runner._dummy_sampler_run(
                hidden_states=self.model_runner._dummy_run(num_tokens=max_num_reqs)
            )

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self):
        return self.model_runner.get_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output,
    ):
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        return
