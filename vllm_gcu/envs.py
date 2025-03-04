import os
from typing import Any, Callable, Dict

environment_variables: Dict[str, Callable[[], Any]] = {
    "VLLM_GCU_DATA_PARALLEL_SIZE": lambda: int(
        os.getenv("VLLM_GCU_DATA_PARALLEL_SIZE", "1")
    ),
    "VLLM_GCU_DATA_PARALLEL_RANK": lambda: int(
        os.getenv("VLLM_GCU_DATA_PARALLEL_RANK", "0")
    ),
    "VLLM_GCU_ENABLE_SEQUENCE_PARALLEL": lambda: bool(
        int(os.getenv("VLLM_GCU_ENABLE_SEQUENCE_PARALLEL", "0"))
    ),
    "VLLM_GCU_ENABLE_EXPERT_PARALLEL": lambda: bool(
        int(os.getenv("VLLM_GCU_ENABLE_EXPERT_PARALLEL", "0"))
    ),
    "VLLM_GCU_HOST_ID": lambda: os.getenv("VLLM_GCU_HOST_ID", "127.0.0.1"),
    "VLLM_GCU_PORT": lambda: int(os.getenv("VLLM_GCU_PORT", 54933)),
    "VLLM_GCU_SAMPLER_ON_CPU": lambda: int(os.getenv("VLLM_GCU_SAMPLER_ON_CPU", "0")),
    "VLLM_GCU_DEBUG_PDONLY": lambda: bool(int(os.getenv("VLLM_GCU_DEBUG_PDONLY", "0"))),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()


def __dir__():
    return list(environment_variables.keys())
