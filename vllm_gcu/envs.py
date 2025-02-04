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
        os.getenv("VLLM_GCU_ENABLE_SEQUENCE_PARALLEL", "1")
    ),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()


def __dir__():
    return list(environment_variables.keys())
