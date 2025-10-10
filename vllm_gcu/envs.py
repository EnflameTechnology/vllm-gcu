import os
from typing import Any, Callable, Dict

environment_variables: Dict[str, Callable[[], Any]] = {
    "VLLM_GCU_ENABLE_SEQUENCE_PARALLEL":
    lambda: bool(int(os.getenv("VLLM_GCU_ENABLE_SEQUENCE_PARALLEL", "0"))),
    "VLLM_GCU_SAMPLER_ON_CPU":
    lambda: int(os.getenv("VLLM_GCU_SAMPLER_ON_CPU", "0")),
    "VLLM_DUMP_SNAPSHOT_EVERY_N_STEP":
    lambda: int(os.getenv("VLLM_DUMP_SNAPSHOT_EVERY_N_STEP", "0")),
    "VLLM_GCU_RANK_LOG_PATH":
    lambda: (None if os.getenv("VLLM_GCU_RANK_LOG_PATH", None) is None else os.
             path.expanduser(os.getenv("VLLM_GCU_RANK_LOG_PATH", "."))),
    "VLLM_GCU_ENABLE_PARALLEL_COMPUTE":
    lambda: bool(int(os.getenv("VLLM_GCU_ENABLE_PARALLEL_COMPUTE", "0"))),
    # may change in GCUPlatform
    "VLLM_GCU_DEEPSEEK_FUSION":
    lambda: bool(int(os.getenv("VLLM_GCU_DEEPSEEK_FUSION", "0"))),
    "VLLM_GCU_ENABLE_COMPILE_DUMP":
    lambda: bool(int(os.getenv("VLLM_GCU_ENABLE_COMPILE_DUMP", "0"))),
    "VLLM_GCU_REJECT_SAMPLER_CHECK":
    lambda: bool(int(os.getenv("VLLM_GCU_REJECT_SAMPLER_CHECK", "0"))),
    "VLLM_GCU_FORCE_EP_BALANCE":
    lambda: bool(int(os.getenv("VLLM_GCU_FORCE_EP_BALANCE", "0"))),
    "VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER":
    lambda: bool(int(os.getenv("VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER", "0"))),
    "VLLM_GCU_HOOKS":
    lambda: None if "VLLM_GCU_HOOKS" not in os.environ else os.environ[
        "VLLM_GCU_HOOKS"].split(","),
    "VLLM_GCU_NET_CONFIG":
    lambda: (None if os.getenv("VLLM_GCU_NET_CONFIG", None) is None else os.
             path.expanduser(os.getenv("VLLM_GCU_NET_CONFIG", "."))),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()


def __dir__():
    return list(environment_variables.keys())
