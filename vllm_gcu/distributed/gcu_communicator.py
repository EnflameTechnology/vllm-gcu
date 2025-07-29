#!/usr/bin/env python
# coding=utf-8
from typing import Optional

import torch
import torch_gcu  # noqa: F401
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase, All2AllManagerBase
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

from vllm_gcu.distributed.pyeccl import PyEcclCommunicator


class GCUAll2AllManager(All2AllManagerBase):

    def __init__(self):
        pass


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
        self.all2all_manager = GCUAll2AllManager()

    def all_reduce(self, input_):
        # always try custom allreduce first,
        # and then pynccl.
        ca_comm = self.ca_comm
        if ca_comm is not None and not ca_comm.disabled and \
            ca_comm.should_custom_ar(input_):
            out = ca_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            if hasattr(torch_gcu.distributed, "all_reduce_outplace"):
                out = torch.empty_like(input_)
                torch_gcu.distributed.all_reduce_outplace(out, input_, group=self.device_group)
            else:
                out = input_.clone()
                torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def dispatch(self, hidden_states, router_logits) -> tuple[torch.Tensor, torch.Tensor]:
        # [TODO] modular kernel
        return hidden_states, router_logits

    def combine(self, hidden_states) -> torch.Tensor:
        # [TODO] modular kernel
        return hidden_states
