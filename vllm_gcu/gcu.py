#!/usr/bin/env python
# coding=utf-8
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch_gcu

import vllm.envs as envs

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms.interface import _Backend, DeviceCapability, Platform, PlatformEnum

logger = init_logger(__name__)


class GCUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "GCU"
    device_type: str = "gcu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "TOPS_VISIBLE_DEVICES"
    simple_compile_backend: str = "topsgraph"

    supported_quantization: list[str] = ["awq", "gptq", "moe_wna16", "fp8"]

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
        if use_v1:
            logger.info("Using Flash Attention backend on V1 engine.")
            raise NotImplementedError
        if use_mla:
            logger.info("Using Triton MLA backend.")
            return "vllm_gcu.attention.backends.mla.GCUMLABackend"
        if selected_backend == _Backend.FLASHINFER:
            logger.info("Using FlashInfer backend.")
            raise NotImplemented
        elif selected_backend == _Backend.XFORMERS:
            logger.info("Using XFormers backend.")
            return "vllm_gcu.attention.backends.xformers.GCUXFormersBackend"
        elif selected_backend:
            raise NotImplemented(f"{selected_backend}")
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
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.gcu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False if enforce_eager else True

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import vllm_gcu.envs as gcu_envs
        import vllm_gcu.kernels

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        is_30 = cls.get_device_capability()[0] == 13

        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                parallel_config.worker_cls = (
                    "vllm_gcu.worker.multi_step_worker.GCUMultiStepWorker"
                )
            elif vllm_config.speculative_config:
                raise NotImplementedError
            else:
                parallel_config.worker_cls = "vllm_gcu.worker.worker.GCUWorker"

        parallel_config.disable_custom_all_reduce = True

        if (
            parallel_config.distributed_executor_backend == "mp"
            and parallel_config.world_size > 1
            and envs.VLLM_WORKER_MULTIPROC_METHOD != "spawn"
        ):
            envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"
            logger.info(
                "Set VLLM_WORKER_MULTIPROC_METHOD='spawn' because others are not supported on GCUs."
            )

        cache_config = vllm_config.cache_config

        if cache_config:
            cache_config.block_size = 64
            if cache_config.cache_dtype == "fp8" and is_30:
                cache_config.cache_dtype = "int8"  # fp8 is not support now

        model_config = vllm_config.model_config
        if model_config.quantization in ["gptq", "awq", "moe_wna16"]:
            model_config.quantization = f"{model_config.quantization}_gcu"
        elif model_config.quantization == "fp8" and is_30:
            model_config.quantization = "w8a8_gcu"

        vllm_config.quant_config = VllmConfig._get_quantization_config(
            model_config, vllm_config.load_config
        )

        scheduler_config = vllm_config.scheduler_config
        if (
            gcu_envs.VLLM_GCU_DATA_PARALLEL_SIZE > 1
            and gcu_envs.VLLM_GCU_ENABLE_EXPERT_PARALLEL
        ):
            # use prioritied scheduling when DP and EP
            scheduler_config.policy = "priority"

        envs.VLLM_NO_USAGE_STATS = True

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        pass

    @classmethod
    def get_current_memory_usage(
        cls, device: Optional[torch.types.Device] = None
    ) -> float:
        torch.gcu.reset_peak_memory_stats(device)
        return torch.gcu.max_memory_allocated(device)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        raise NotImplementedError
