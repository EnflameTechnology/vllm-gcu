import torch
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe


class PyEcclPipe(PyNcclPipe):
    def _select_device(self, device: str):
        if device == "gcu":
            return torch.device(f"gcu:{self.local_rank}")
        else:
            return torch.device("cpu")
