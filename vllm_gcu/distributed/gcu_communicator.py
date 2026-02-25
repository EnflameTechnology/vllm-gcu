#!/usr/bin/env python
# coding=utf-8
from typing import Optional, Union

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
from vllm.platforms import current_platform

import vllm_gcu.envs as gcu_envs
from vllm_gcu.distributed.pyeccl import PyEcclCommunicator

logger = init_logger(__name__)


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
        self.use_ep = False
        if config is not None:
            self.use_etp = config.parallel_config.data_parallel_size > 1 \
                and not config.parallel_config.enable_expert_parallel
            self.use_ep = config.parallel_config.enable_expert_parallel \
                and (config.parallel_config.data_parallel_size > 1
                     or gcu_envs.VLLM_GCU_ENABLE_SEQUENCE_PARALLEL)

        if "ep" in unique_name:
            all2all_backend = envs.VLLM_ALL2ALL_BACKEND
            if self.world_size == 1:
                self.all2all_manager = GCUAll2AllManager()
            elif self.use_ep:
                if all2all_backend == "deepep_high_throughput":
                    from vllm.distributed.device_communicators.all2all import DeepEPHTAll2AllManager
                    self.all2all_manager = DeepEPHTAll2AllManager(self.cpu_group)
                    logger.info("Using DeepEP High-Throughput all2all manager.")
                elif all2all_backend == "deepep_low_latency":
                    from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager
                    self.all2all_manager = DeepEPLLAll2AllManager(self.cpu_group)
                    logger.info("Using DeepEP Low-Latency all2all manager.")
                else:
                    self.all2all_manager = GCUAll2AllManager()
            elif self.use_etp:
                if all2all_backend == "naive":
                    self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
                    logger.info("Using naive all2all manager.")
                elif all2all_backend == "allgather_reducescatter":
                    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
                    self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                    logger.info("Using AllGather-ReduceScatter all2all manager.")
                else:
                    raise ValueError(f"Unknown all2all backend: {all2all_backend}")
            else:
                # EP=False && DP==1
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

    def dispatch(
            self,
            hidden_states,
            router_logits,
            is_sequence_parallel: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # a bit tricky, moe layer call dispatch/combine when dp>1.
        # but we only need it when not ep.
        if self.use_etp:
            assert self.all2all_manager is not None
            hidden_states, router_logits = self.all2all_manager.dispatch(
                hidden_states, router_logits, is_sequence_parallel)
        return hidden_states, router_logits

    def combine(self,
                hidden_states,
                is_sequence_parallel: bool = False) -> torch.Tensor:
        if self.use_etp:
            assert self.all2all_manager is not None
            hidden_states = self.all2all_manager.combine(hidden_states,
                                                         is_sequence_parallel)
        return hidden_states

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        if self.pynccl_comm is not None and not self.pynccl_comm.disabled:
            self.pynccl_comm.reduce_scatter(output, input_tensor)
        else:
            torch.distributed.reduce_scatter_tensor(
                output, input_tensor, group=self.device_group
            )

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(self,
                        input_: torch.Tensor,
                        dim: int = -1,
                        sizes: Optional[list[int]] = None):
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_.shape[0] == sum(sizes)
            return torch.ops.vllm.reduce_scatter_v(input_, sizes, self.unique_name)
        else:
            assert input_.shape[0] % world_size == 0
            return self.reduce_scatter(input_)

    def all_gatherv(self,
                    input_: Union[torch.Tensor, list[torch.Tensor]],
                    dim: int = 0,
                    sizes: Optional[list[int]] = None):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        def _all_gather_single(input_: torch.Tensor,
                               sizes: Optional[list[int]] = None):
            if sizes is not None:
                return torch.ops.vllm.all_gather_v(input_, sizes, self.unique_name)
            else:
                return self.all_gather(input_)

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))

        return output_list