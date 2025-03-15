from typing import Optional, Union

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from vllm_gcu.distributed.pyeccl_wrapper import (
    buffer_type,
    ecclComm_t,
    ecclDataTypeEnum,
    ECCLLibrary,
    ecclRedOpTypeEnum,
    ecclUniqueId,
    topsStream_t,
)

logger = init_logger(__name__)


class PyEcclCommunicator:

    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyEcclCommunicator to. If None,
                it will be bind to f"gcu:{local_rank}".
            library_path: the path to the ECCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert (
                dist.get_backend(group) != dist.Backend.ECCL
            ), "PyEcclCommunicator should be attached to a non-ECCL group."
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return
        try:
            self.eccl = ECCLLibrary(library_path)
        except Exception:
            # disable because of missing ECCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        logger.info("vLLM is using eccl==%s", self.eccl.ecclGetVersion())

        if self.rank == 0:
            # get the unique id from ECCL
            self.unique_id = self.eccl.ecclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = ecclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"gcu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # eccl communicator and stream will use this device
        # `torch.gcu.device` is a context manager that changes the
        # current gcu device to the specified one
        with torch.gcu.device(device):
            self.comm: ecclComm_t = self.eccl.ecclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

            stream = torch.gcu.current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

        # TODO something error when pyeccl with graph, seems related to stream
        self.disabled = True

    def all_reduce(
        self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None
    ) -> torch.Tensor:
        if self.disabled:
            return None
        # eccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this eccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        # out_tensor = torch.empty_like(in_tensor)
        out_tensor = in_tensor.clone()

        if stream is None:
            stream = torch.gcu.current_stream()
        self.eccl.ecclAllReduce(
            buffer_type(out_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            ecclDataTypeEnum.from_torch(in_tensor.dtype),
            ecclRedOpTypeEnum.from_torch(op),
            self.comm,
            topsStream_t(stream.gcu_stream),
        )
        return out_tensor

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        # eccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this eccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = torch.gcu.current_stream()
        self.eccl.ecclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ecclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            topsStream_t(stream.gcu_stream),
        )

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # eccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this eccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = torch.gcu.current_stream()
        self.eccl.ecclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            ecclDataTypeEnum.from_torch(input_tensor.dtype),
            ecclRedOpTypeEnum.from_torch(op),
            self.comm,
            topsStream_t(stream.gcu_stream),
        )

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this eccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = torch.gcu.current_stream()
        self.eccl.ecclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ecclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            topsStream_t(stream.gcu_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this eccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = torch.gcu.current_stream()
        self.eccl.ecclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ecclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            topsStream_t(stream.gcu_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this eccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = torch.gcu.current_stream()
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # ECCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.eccl.ecclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ecclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            topsStream_t(stream.gcu_stream),
        )
