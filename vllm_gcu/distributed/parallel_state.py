from typing import Optional

import torch
from vllm.distributed.parallel_state import (
    all_reduce,
    GroupCoordinator,
    init_model_parallel_group,
)
from vllm.utils import vllm_lib

from vllm_gcu.distributed.pyeccl import PyEcclCommunicator


vllm_lib.impl("all_reduce", all_reduce, dispatch_key="PrivateUse1")


class GroupWrapper:
    # proxy
    def __init__(self, group: GroupCoordinator):
        self.group = group
        self.group.device = torch.device(f"gcu:{self.group.local_rank}")
        self.group.pynccl_comm = PyEcclCommunicator(
            group=self.group.cpu_group, device=self.group.device
        )

    def __getattr__(self, method: str):
        return getattr(self.group, method)

    def all_to_all_single(
        self, output, input, output_split_sizes=None, input_split_sizes=None
    ):
        return torch.distributed.all_to_all(
            output, input, output_split_sizes, input_split_sizes
        )


_DP = None


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None
    return _DP


get_data_parallel_group = get_dp_group


def initialize_data_parallel(
    data_parallel_size: int,
    global_world_size,
    unique_rank,
    backend: Optional[str] = None,
):
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_data_parallel_groups: int = world_size // data_parallel_size

    global _DP
    assert _DP is None, "data parallel group is already initialized"

    group_ranks = []
    for i in range(num_data_parallel_groups):
        ranks = list(range(i, world_size, num_data_parallel_groups))
        group_ranks.append(ranks)

    _DP = GroupWrapper(
        init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dp"
        )
    )
