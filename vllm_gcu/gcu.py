#!/usr/bin/env python
# coding=utf-8
import hashlib
import os
import random
import types
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from vllm.config import SupportsHash
from vllm.platforms.interface import (
    _Backend,
    CpuArchEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)


class AdditionalConfig(dict, SupportsHash):
    def compute_hash(self) -> str:
        return str(hash(frozenset(self.items())))


class GCUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "GCU"
    device_type: str = "gcu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "TOPS_VISIBLE_DEVICES"
    simple_compile_backend: str = "topsgraph"
    supported_quantization: List[str] = [
        "awq_gcu",
        "gptq_gcu",
        "moe_wna16",
        "moe_wna16_gcu",
        "w8a8_gcu",
        "fp8",
        "fp8_gcu",
    ]

    def is_cuda_alike(self) -> bool:
        return True

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
    ) -> str:
        if use_mla:
            if use_v1:
                return "vllm_gcu.v1.attention.mla.GCUMLABackend"
            else:
                return "vllm_gcu.attention.backends.mla.GCUMLABackend"
        if use_v1:
            return "vllm_gcu.v1.attention.flash_attn.GCUFlashAttentionBackend"
        if selected_backend == _Backend.FLASHINFER:
            raise NotImplementedError
        elif selected_backend == _Backend.XFORMERS:
            return "vllm_gcu.attention.backends.xformers.GCUXFormersBackend"
        elif selected_backend == _Backend.FLASH_ATTN:
            return "vllm_gcu.attention.backends.flash_attn.FlashAttentionBackend"
        elif selected_backend:
            raise NotImplementedError(f"{selected_backend}")
        return "vllm_gcu.attention.backends.xformers.GCUXFormersBackend"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.gcu.get_device_capability(device_id)
        return DeviceCapability(major=major + 10, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def has_device_capability(
        cls, capability: Union[Tuple[int, int], int], device_id: int = 0
    ) -> bool:
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False

        if isinstance(capability, tuple):
            return current_capability >= capability

        return current_capability.to_int() >= capability

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.gcu.get_device_name(device_id)

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_uuid(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.gcu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False if enforce_eager else True

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        import tops_extension.torch  # noqa: F401
        import torch_gcu  # noqa: F401
        import torch_gcu.transfer_to_gcu  # noqa: F401

        import vllm_gcu.compilation  # noqa: F401
        import vllm_gcu.distributed  # noqa: F401
        import vllm_gcu.kernels  # noqa: F401

        if parser:
            key = "--disable-async-output-proc"
            if key in parser._option_string_actions:
                # set disable_async_output_proc default True
                parser._option_string_actions[key].default = True

            key = "--device"
            if key in parser._option_string_actions:
                # set "gcu" to device
                parser._option_string_actions[key].choices += ["gcu"]

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        import vllm.envs as envs

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        compilation_config = vllm_config.compilation_config

        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                parallel_config.worker_cls = (
                    "vllm_gcu.worker.multi_step_worker.GCUMultiStepWorker"
                )
            elif vllm_config.speculative_config:
                parallel_config.worker_cls = (
                    "vllm_gcu.worker.spec_decode.spec_decode_worker.create_spec_worker"
                )
                parallel_config.sd_worker_cls = "vllm_gcu.worker.worker.GCUWorker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = "vllm_gcu.v1.worker.gcu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm_gcu.worker.worker.GCUWorker"

        if envs.VLLM_USE_V1 and torch.__version__.startswith("2.5.1"):

            def stateless_init_dp_group(self):
                from vllm_gcu.distributed.utils import (
                    stateless_init_torch_distributed_process_group,
                )

                # use gloo since the engine process might not have cuda device
                dp_group = stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    self.get_next_dp_init_port(),
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                )

                return dp_group

            parallel_config.stateless_init_dp_group = types.MethodType(
                stateless_init_dp_group, parallel_config
            )

        # Force disable custom all reduce
        parallel_config.disable_custom_all_reduce = True
        if (
            parallel_config.distributed_executor_backend == "mp"
            and parallel_config.world_size > 1
        ):
            # force spawn multiprocessing method as others not support
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"

        if cache_config:
            if cache_config.block_size is None:
                # set block size to 64 for gcu if not specific
                cache_config.block_size = 64

            if (
                cache_config.cache_dtype.startswith("fp8")
                and cls.get_device_capability() == 130
            ):
                cache_config.cache_dtype = "int8"

        if (
            parallel_config.data_parallel_size > 1
            and parallel_config.enable_expert_parallel
            and scheduler_config.policy == "priority"
        ):
            # use prioritied scheduling when DP and EP
            scheduler_config.scheduler_cls = (
                "vllm_gcu.scheduler.PriorityScheduler"  # priority to preempt
            )

        if compilation_config:
            if compilation_config.level > 0:
                compilation_config.backend = "topsgraph"

            if compilation_config.level == 3:
                os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
                envs.VLLM_DISABLE_COMPILE_CACHE = True
                # TODO: WA for bug in vllm, to be removed after 0.8.2
                if not compilation_config.cache_dir:
                    factors = []
                    config_hash = vllm_config.compute_hash()
                    factors.append(config_hash)

                    hash_key = hashlib.md5(str(factors).encode()).hexdigest()[:10]

                    cache_dir = os.path.join(
                        envs.VLLM_CACHE_ROOT,
                        "torch_compile_cache",
                        hash_key,
                    )
                    compilation_config.cache_dir = cache_dir

                os.makedirs(compilation_config.cache_dir, exist_ok=True)

                world_size = vllm_config.parallel_config.world_size
                dp_size = vllm_config.parallel_config.data_parallel_size
                for rank in range(world_size):
                    for dp_rank in range(dp_size):
                        local_cache_dir = os.path.join(
                            compilation_config.cache_dir, f"rank_{rank}_{dp_rank}"
                        )
                        os.makedirs(local_cache_dir, exist_ok=True)

                # TODO: remove after rmsnorm pattern fix in official.
                compilation_config.pass_config.enable_fusion = False
                compilation_config.pass_config.dump_graph_stages.extend(
                    ["before_fusion", "after_pattern_match", "after_fusion"]
                )
                compilation_config.custom_ops = ["all"]

        if model_config:
            model_config.enable_sleep_mode = False
            if envs.VLLM_USE_V1:
                model_config.use_async_output_proc = True

        additional_config = vllm_config.additional_config
        if additional_config is None:
            # make sure additional_config is not None
            vllm_config.additional_config = AdditionalConfig({})
        vllm_config.additional_config.update({"all_dp_in_decode": False})

        # Disable usage status for security
        envs.VLLM_NO_USAGE_STATS = "1"

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        not_supported_model_archs = []
        if model_arch in not_supported_model_archs:
            raise NotImplementedError(model_arch)

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        if quant not in cls.supported_quantization:
            raise NotImplementedError(quant)

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        return Platform.get_cpu_architecture()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return True

    @classmethod
    def get_current_memory_usage(
        cls, device: Optional[torch.types.Device] = None
    ) -> float:
        torch.gcu.reset_peak_memory_stats(device)
        return torch.gcu.max_memory_allocated(device)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_gcu.distributed.gcu_communicator.GCUCommunicator"

    @classmethod
    def supports_fp8(cls) -> bool:
        current_capability = cls.get_device_capability(device_id=0)
        if current_capability is None:
            return False

        return current_capability.to_int() >= 140

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        return False

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn
