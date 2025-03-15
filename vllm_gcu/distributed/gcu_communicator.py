#!/usr/bin/env python
# coding=utf-8
from typing import Optional

import torch
import torch_gcu  # noqa: F401
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

from vllm_gcu.distributed.pyeccl import PyEcclCommunicator


class GCUCommunicator(CudaCommunicator):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: Optional[torch.device] = None,
        device_group: Optional[ProcessGroup] = None,
        unique_name: str = "",
    ):
        DeviceCommunicatorBase.__init__(
            self, cpu_group, device, device_group, unique_name
        )

        self.use_custom_allreduce = False
        self.use_pynccl = True

        self.pynccl_comm = PyEcclCommunicator(
            group=self.cpu_group,
            device=torch.device(
                f"gcu:{torch.distributed.get_rank() % torch.gcu.device_count()}"
            ),
        )
        self.ca_comm = None
