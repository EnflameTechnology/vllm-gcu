#!/usr/bin/env python
# coding=utf-8
import random
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.platforms.interface import (
    _Backend,
    CpuArchEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from vllm.utils import FlexibleArgumentParser


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
    ]

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
                raise NotImplementedError
                # return "vllm_gcu.v1.attention.backends.mla.GCUMLABackend"
            else:
                return "vllm_gcu.attention.backends.mla.GCUMLABackend"
        if use_v1:
            raise NotImplementedError
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
    def pre_register_and_update(
        cls, parser: Optional[FlexibleArgumentParser] = None
    ) -> None:
        import tops_extension.torch  # noqa: F401
        import torch_gcu  # noqa: F401
        import torch_gcu.transfer_to_gcu  # noqa: F401

        import vllm_gcu.distributed  # noqa: F401
        import vllm_gcu.kernels  # noqa: F401

        if parser:
            key = "--kv-cache-dtype"
            if key in parser._option_string_actions:
                # add kv cache dtype: int8
                parser._option_string_actions[key].choices += ["int8"]

            key = "--disable-async-output-proc"
            if key in parser._option_string_actions:
                # set disable_async_output_proc default True
                parser._option_string_actions[key].default = True

            key = "--device"
            if key in parser._option_string_actions:
                # set "gcu" to device
                parser._option_string_actions[key].choices += ["gcu"]

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import vllm.envs as envs

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                parallel_config.worker_cls = (
                    "vllm_gcu.worker.multi_step_worker.GCUMultiStepWorker"
                )
            elif vllm_config.speculative_config:
                parallel_config.worker_cls = (
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                )
                parallel_config.sd_worker_cls = "vllm_gcu.worker.worker.GCUWorker"
            else:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError
                    # parallel_config.worker_cls = (
                    #     "vllm_gcu.v1.worker.gcu_worker.GCUWorker"
                    # )
                else:
                    parallel_config.worker_cls = "vllm_gcu.worker.worker.GCUWorker"

        # Force disable custom all reduce
        parallel_config.disable_custom_all_reduce = True
        if (
            parallel_config.distributed_executor_backend == "mp"
            and parallel_config.world_size > 1
        ):
            # force spawn multiprocessing method as others not support
            import os
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"

        if cache_config and cache_config.block_size is None:
            # set block size to 64 for gcu if not specific
            cache_config.block_size = 64

        if (
            parallel_config.data_parallel_size > 1
            and parallel_config.enable_expert_parallel
            and scheduler_config.policy == "priority"
        ):
            # use prioritied scheduling when DP and EP
            scheduler_config.scheduler_cls = (
                "vllm_gcu.scheduler.PriorityScheduler"  # priority to preempt
            )

        if model_config:
            model_config.enable_sleep_mode = False

        additional_config = vllm_config.additional_config
        if additional_config is None:
            # make sure additional_config is not None
            vllm_config.additional_config = {}
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

        return current_capability.to_int() >= 40

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        return False

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn
