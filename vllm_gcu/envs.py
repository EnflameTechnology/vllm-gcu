import os
from typing import Any, Callable, Dict

environment_variables: Dict[str, Callable[[], Any]] = {
    "VLLM_GCU_ENABLE_SEQUENCE_PARALLEL": lambda: bool(
        int(os.getenv("VLLM_GCU_ENABLE_SEQUENCE_PARALLEL", "0"))
    ),
    "VLLM_GCU_SAMPLER_ON_CPU": lambda: int(os.getenv("VLLM_GCU_SAMPLER_ON_CPU", "0")),
    "VLLM_DUMP_SNAPSHOT_EVERY_N_STEP": lambda: int(
        os.getenv("VLLM_DUMP_SNAPSHOT_EVERY_N_STEP", 0)
    ),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()


def __dir__():
    return list(environment_variables.keys())
