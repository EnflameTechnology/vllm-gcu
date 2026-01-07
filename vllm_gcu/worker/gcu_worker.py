from typing import Union
from importlib.util import find_spec
import os
import gc
import contextlib
import torch
import torch_gcu
from vllm.utils import MemorySnapshot, GiB_bytes
from vllm.model_executor import set_random_seed
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm_gcu.worker.gcu_model_runner import GCUModelRunner
import vllm_gcu.envs as gcu_envs
import vllm.envs as envs
from vllm_gcu.utils import get_tx_ctx
from vllm.utils import cdiv
import orjson


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
        import vllm_gcu.envs as gcu_envs

        if gcu_envs.VLLM_GCU_RANK_LOG_PATH:
            # before init dist, since we want to split eccl init logs
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            world_size = vllm_config.parallel_config.world_size
            rank_across_dp = dp_rank * world_size + rank
            rank_log = os.path.join(gcu_envs.VLLM_GCU_RANK_LOG_PATH,
                                    f"worker_{rank_across_dp}.log")
            with open(rank_log, "w", buffering=1) as f:
                os.dup2(f.fileno(), 1)
                os.dup2(f.fileno(), 2)
        if vllm_config.additional_config.get('set_cpu_affinity', False):
            current_platform.set_cpu_affinity(local_rank)
        
        self.use_async_scheduling = vllm_config.additional_config.get('async_scheduling', None)
        
        self.enable_fuse_mtp = vllm_config.additional_config.get('deepseek_fused_mtp', False)

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
        self.requested_memory = (self.init_snapshot.total_memory *
                                 self.cache_config.gpu_memory_utilization)

        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)  # noqa: E731
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes.")

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        if self.enable_fuse_mtp:
            from vllm_gcu.worker.gcu_fuse_mtp_model_runner import FuseMTPGCUModelRunner
            self.model_runner: FuseMTPGCUModelRunner = FuseMTPGCUModelRunner(
                self.vllm_config, self.device)
        else:
            self.model_runner: GCUModelRunner = GCUModelRunner(
                self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output,
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        message = f"execute_{num_scheduled_tokens}"
        color = "green"
        domain = "VLLM"
        category = "execute"
        payload_str = None

        if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
            block_size = self.vllm_config.cache_config.block_size

            num_blocks = {}
            for i, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
                seq_len = scheduler_output.scheduled_cached_reqs.num_computed_tokens[i] + num_scheduled_tokens[req_id]
                num_blocks[req_id] = cdiv(seq_len, block_size)

            for new_req_data in scheduler_output.scheduled_new_reqs:
                req_id = new_req_data.req_id
                seq_len = new_req_data.num_computed_tokens + num_scheduled_tokens[req_id]
                num_blocks[req_id] = cdiv(seq_len, block_size)


            payload = {"request_info": []}
            for req_id, num_token in num_scheduled_tokens.items():
                num_block = -1
                if req_id in num_blocks:
                    num_block = num_blocks[req_id]

                payload["request_info"].append({
                    "req_id": req_id,
                    "num_token": num_token,
                    "num_block": num_block
                })

            payload_str = orjson.dumps(payload)

        tx_ctx = get_tx_ctx(message, color, domain, category, payload_str)
        with tx_ctx:
            return super().execute_model(scheduler_output)

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(0, uniform_decode = True)


    def compile_or_warm_up_model(self) -> None:
        super().compile_or_warm_up_model()
        if self.model_runner.eplb_state is not None:
            self.model_runner.eplb_state.expert_load_pass.zero_()