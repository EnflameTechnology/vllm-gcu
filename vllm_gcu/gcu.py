#!/usr/bin/env python
# coding=utf-8
import os
import random
import types
from functools import lru_cache, wraps
from typing import List, Optional, Tuple, Union, Callable, TypeVar, TYPE_CHECKING
from typing_extensions import ParamSpec

import numpy as np
import torch
import vllm.envs as envs
from vllm.platforms.interface import (
    _Backend,
    CpuArchEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from vllm.logger import init_logger

import vllm_gcu.envs as gcu_envs

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

def with_efml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        import pyefml
        pyefml.efmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pyefml.efmlShutdown()

    return wrapper


class GCUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "GCU"
    device_type: str = "gcu"
    dispatch_key: str = "PrivateUse1"
    dist_backend: str = "eccl"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "TOPS_VISIBLE_DEVICES"
    simple_compile_backend: str = "topsgraph"
    supported_quantization: List[str] = [
        "awq_gcu",
        "gptq_gcu",
        "moe_wna16",
        "moe_wna16_gcu",
        "w8a8_gcu",
        "w4a8",
        "w4a8_gcu",
        "fp8",
        "fp8_gcu",
        "compressed-tensors",
    ]

    def is_cuda_alike(self) -> bool:
        return True

    def get_vit_attn_backend(cls, head_size: int, dtype: torch.dtype) -> _Backend:
        return _Backend.FLASH_ATTN

    @classmethod
    def opaque_attention_op(cls) -> bool:
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
        has_sink: bool,
        use_sparse: bool,
    ) -> str:
        if use_mla:
            if use_sparse:
                logger.info_once("Using Sparse MLA backend on V1 engine.")
                if gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
                    return ("vllm_gcu.attention.backends.flashmla_sparse_fusion."
                            "FlashMLASparseFusionBackend")
                else:
                    return ("vllm_gcu.attention.backends.flashmla_sparse."
                            "FlashMLASparseBackend")
            if selected_backend == _Backend.FLASHMLA:
                raise ValueError("FLASHMLA is not supported on GCU yet!")
                # return "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
            if gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
                return "vllm_gcu.attention.backends.mla_v1_fusion.GCUMLAFusionBackend"
            else:
                return "vllm_gcu.attention.backends.mla_v1.GCUMLABackend"

        return "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"


    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.gcu.get_device_capability(device_id)
        return DeviceCapability(major=major + 10, minor=minor)

    @classmethod
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
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.gcu.set_device(device)

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        import tops_extension.torch  # noqa: F401
        import torch_gcu  # noqa: F401
        import torch_gcu.transfer_to_gcu  # noqa: F401

        import vllm_gcu.kernels  # noqa: F401
        import vllm_gcu.compilation  # noqa: F401
        import vllm_gcu.distributed  # noqa: F401
        import vllm_gcu.patch  # noqa: F401

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        if parser:
            key = "--device"
            if key in parser._option_string_actions:
                # set "gcu" to device
                parser._option_string_actions[key].choices += ["gcu"]
                parser._option_string_actions[key].default = "gcu"

            key = "--block-size"
            if key in parser._option_string_actions:
                parser._option_string_actions[key].choices += [256]

            key = "--kv-cache-dtype"
            if key in parser._option_string_actions:
                parser._option_string_actions[key].choices += ['fp8_ds_mla', 'int8']

            # key = "--disable-async-output-proc"
            # if key in parser._option_string_actions:
            #     # set disable_async_output_proc default True
            #     parser._option_string_actions[key].default = True

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        compilation_config = vllm_config.compilation_config
        speculative_config = vllm_config.speculative_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_gcu.worker.gcu_worker.GCUWorker"

        distributed_executor_backend = parallel_config.distributed_executor_backend
        if isinstance(distributed_executor_backend, str):
            if distributed_executor_backend == "mp":
                from vllm_gcu.executor.executor import GCUMultiprocExecutor

                parallel_config.distributed_executor_backend = GCUMultiprocExecutor
            elif distributed_executor_backend == "ray":
                pass

        # Force disable custom all reduce
        parallel_config.disable_custom_all_reduce = True
        if (
            parallel_config.distributed_executor_backend == "mp"
        ):
            from vllm_gcu.executor.executor import GCUMultiprocExecutor
            # force spawn multiprocessing method as others not support
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"

            parallel_config.distributed_executor_backend = GCUMultiprocExecutor

        if cache_config:
            if cache_config.block_size is None:
                # set block size to 64 for gcu if not specific
                cache_config.block_size = 64

        enable_deepseek_fused_mtp = gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION
        if not enable_deepseek_fused_mtp and \
            vllm_config.additional_config.get("deepseek_fused_mtp", False):
            enable_deepseek_fused_mtp = True
            os.environ["VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION"] = "1"
        if enable_deepseek_fused_mtp:
            logger.info("Using deepseek fused mtp")
            if vllm_config.additional_config.get("deepseek_fused_mtp_use_penalty", True):
                vllm_config.additional_config.setdefault("deepseek_fused_mtp_penalty_max_prompt_len", 512)
                vllm_config.additional_config.setdefault("deepseek_fused_mtp_penalty_max_output_len", 1024)
                
        vllm_config.additional_config.update({"deepseek_fused_mtp": enable_deepseek_fused_mtp})

        if compilation_config:
            if compilation_config.level > 0:
                compilation_config.backend = "topsgraph"

            if compilation_config.level == 3:
                compilation_config.pass_config.enable_noop = False
                compilation_config.pass_config.enable_sequence_parallelism = False
                compilation_config.pass_config.enable_fi_allreduce_fusion = False
                compilation_config.pass_config.enable_fusion = False
                compilation_config.pass_config.enable_attn_fusion = False

                compilation_config.custom_ops = ["all"]

            if vllm_config.parallel_config.data_parallel_size > 1:
                compilation_config.compile_sizes.append(0)
                compilation_config.cudagraph_capture_sizes.append(0)  # capture 0 graph

            if enable_deepseek_fused_mtp:
                assert vllm_config.speculative_config is not None
                spec_k = vllm_config.speculative_config.num_speculative_tokens
                cudagraph_capture_sizes = compilation_config.cudagraph_capture_sizes.copy()
                new_cudagraph_capture_sizes = []
                uniform_batch = True
                for s in cudagraph_capture_sizes:
                    if s % (1 + spec_k) != 0:
                        uniform_batch = False
                    new_cudagraph_capture_sizes.append(s * (1 + spec_k))
                if not uniform_batch:
                    compilation_config.cudagraph_capture_sizes = new_cudagraph_capture_sizes
                    compilation_config.init_with_cudagraph_sizes(new_cudagraph_capture_sizes)
                logger.info(f'for mtp_fusion, capture sizes is {compilation_config.cudagraph_capture_sizes}')

        if model_config:
            model_config.enable_sleep_mode = False

        additional_config = vllm_config.additional_config
        async_scheduling = additional_config.get("async_scheduling", None)
        if async_scheduling is not None:
            scheduler_config.async_scheduling = async_scheduling
            logger.info("override async_scheduling of scheduler_config by additional_config")

        if additional_config.get("tokenizer_mode", False):
            model_config.tokenizer_mode = additional_config['tokenizer_mode']

        if scheduler_config.async_scheduling:
            if speculative_config is not None:
                scheduler_config.scheduler_cls = "vllm_gcu.core.scheduler.AsyncScheduler"
        else:
            scheduler_config.scheduler_cls = "vllm_gcu.core.scheduler.GCUScheduler"

        if additional_config.get("enable_eplb", False):
            parallel_config.enable_eplb = True
        num_redundant_experts = additional_config.get("num_redundant_experts", 0)
        if num_redundant_experts > 0:
            assert parallel_config.enable_eplb, "EPLB must be enabled"
            parallel_config.eplb_config.num_redundant_experts = num_redundant_experts

        if "VLLM_GCU_DEEPSEEK_FUSION" not in os.environ and \
            model_config and model_config.hf_text_config.model_type in \
                    ('deepseek_v3', 'deepseek_mtp'):
            os.environ["VLLM_GCU_DEEPSEEK_FUSION"] = "1"
        if (
            enable_deepseek_fused_mtp
            and model_config
            and model_config.hf_text_config.model_type == "deepseek_v32"
        ):
            logger.warning('Disable deepseek fusion ops as not support deepseek_v32 with deepseek_fused_mtp')
            os.environ["VLLM_GCU_DEEPSEEK_FUSION"] = "0"

        # Disable usage status for security
        envs.VLLM_NO_USAGE_STATS = "1"
        if gcu_envs.VLLM_GCU_DEEPSEEK_FUSION:
            logger.info("Deepseek fusion ops enabled.")
        if gcu_envs.VLLM_GCU_ENABLE_PARALLEL_COMPUTE:
            logger.info("Overlap shared experts with dispatch enabled.")

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
        return "vllm_gcu.lora.punica_gcu.PunicaWrapperGCU"

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

    def is_sleep_mode_available(self) -> bool:
        return False

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    model_config: "ModelConfig") -> bool:
        fp8_attention = kv_cache_dtype.startswith("fp8")
        if fp8_attention and not cls.supports_fp8():
            return False
        return True

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        if torch_dtype in [torch.float8_e4m3fn] and not cls.supports_fp8():
            return False
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @with_efml_context
    def set_cpu_affinity(cls, device_id: int) -> None:
        """
        Set CPU affinity for the current process based on GPU device ID.
        """
        import pyefml
        try:
            import psutil
        except ImportError:
            logger.warning(
                "psutil is not available. Cannot set CPU affinity. "
                "Install psutil to enable NUMA affinity optimization.")
            return

        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pyefml.efmlDeviceGetHandleByIndex(physical_device_id)

            pci_info = pyefml.efmlDeviceGetPciInfo(handle)
            pci_busid = pci_info.busId

            net_config = gcu_envs.VLLM_GCU_NET_CONFIG
            if net_config and os.path.exists(net_config):
                import json
                with open(net_config, "r") as f:
                    net_cfgs = json.load(f)

                net_devices = net_cfgs.get(f"{pci_busid[5:].decode()}", None)
                if net_devices:
                    os.environ["UCX_NET_DEVICES"] = ",".join(net_devices)

            device_count = pyefml.efmlDeviceGetCount()

            # Get CPU affinity for this GPU
            # We need to determine the CPU set size first
            cpu_count = os.cpu_count()
            if cpu_count is None:
                logger.warning(
                    "Cannot determine CPU count. Skipping CPU affinity setting."
                )
                return

            cpu_set_size = (cpu_count + 63) // 64

            # Get CPU affinity from EFML
            cpu_affinity_mask = pyefml.efmlDeviceGetCpuAffinity(
                handle, cpu_set_size)

            # Convert the bitmask to a list of CPU IDs
            cpu_ids = []
            for i, mask in enumerate(cpu_affinity_mask):
                for bit in range(64):
                    cpu_id = i * 64 + bit
                    if cpu_id >= cpu_count:
                        break
                    if mask & (1 << bit):
                        cpu_ids.append(cpu_id)

            if cpu_ids:
                # Set CPU affinity using psutil
                current_process = psutil.Process()
                current_process.cpu_affinity(cpu_ids)
                logger.info(
                    "Set CPU affinity for process %d to " \
                    "CPUs %s for logical GCU devices %s, physical_device_id %s pci busid %s",
                    current_process.pid, cpu_ids, device_id, physical_device_id, pci_busid)
            else:
                logger.warning(
                    "No CPU affinity information available for GCU devices %s",
                    device_id)

        except Exception as e:
            logger.warning("Failed to set CPU affinity for GCU devices %s: %s",
                           device_id, str(e))

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """
        Returns if the graph mode is supported by the current platform.
        """
        return True

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """
        Returns a mapping from device_type to a tuple of supported
        kv_buffer_device for nixl.
        """
        return {"gcu": ("gcu", "cuda")}

    @classmethod
    def get_nixl_memory_type(cls) -> Optional[str]:
        """
        Returns the nixl memory type for the current platform.
        """
        return "VRAM"
