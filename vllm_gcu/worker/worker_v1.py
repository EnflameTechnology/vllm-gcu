import torch
from unittest.mock import patch
import os
import gc

# import vllm.device_allocator
from vllm.utils import MemorySnapshot, GiB_bytes
from vllm.model_executor import set_random_seed
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.utils import report_usage_stats
from vllm_gcu import gcumem


class GCUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self.input_ids.to(torch.int64)


with patch("vllm.device_allocator", "cumem", gcumem):
    from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment


class GCUWorker(Worker):
    def __init__(self, *args, **kwargs):
        import vllm_gcu.kernels  # noqa: F401
        import vllm_gcu.compilation  # noqa: F401
        import vllm_gcu.patch  # noqa: F401

        super().__init__(*args, **kwargs)

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
