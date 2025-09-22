#!/usr/bin/env python
# coding=utf-8
from typing import Optional

import torch
import torch_gcu  # noqa: F401
from torch.distributed import ProcessGroup

from vllm.distributed import get_dp_group
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase, All2AllManagerBase
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.all2all import NaiveAll2AllManager
import vllm.envs as envs
from vllm.logger import init_logger

from vllm_gcu.distributed.pyeccl import PyEcclCommunicator

logger = init_logger(__name__)

class AllgathervNaiveManager(NaiveAll2AllManager):

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)

        recvcounts = torch.diff(cu_tokens_across_dp_cpu, dim=0).tolist()
        recvcounts.insert(0, cu_tokens_across_dp_cpu[0].item())
        return torch.ops.vllm.all_gather_v(x, recvcounts,
                                           get_dp_group().unique_name)


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

        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        self.use_etp = False
        if config is not None:
            self.use_etp = config.parallel_config.data_parallel_size > 1 \
                and not config.parallel_config.enable_expert_parallel

        if "ep" in unique_name:
            if self.use_etp:
                all2all_backend = envs.VLLM_ALL2ALL_BACKEND
                if all2all_backend == "naive":
                    self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
                    logger.info("Using naive all2all manager.")
                elif all2all_backend == "allgatherv":
                    self.all2all_manager = AllgathervNaiveManager(self.cpu_group)
                    logger.info("Using allgatherv naive all2all manager.")
                else:
                    raise ValueError(f"Unknown all2all backend: {all2all_backend}")
            else:
                # (EP=False && DP==1) || EP=True
                self.all2all_manager = GCUAll2AllManager()

        # Always init_prepare_finalize.
        self.use_all2all = "ep" in unique_name

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
        # a bit tricky, moe layer call dispatch/combine when dp>1.
        # but we only need it when not ep.
        if self.use_etp:
            assert self.all2all_manager is not None
            hidden_states, router_logits = self.all2all_manager.dispatch(
                hidden_states, router_logits)
        return hidden_states, router_logits

    def combine(self, hidden_states) -> torch.Tensor:
        if self.use_etp:
            assert self.all2all_manager is not None
            hidden_states = self.all2all_manager.combine(hidden_states)
        return hidden_states
