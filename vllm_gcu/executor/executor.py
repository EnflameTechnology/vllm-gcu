#!/usr/bin/env python
# coding=utf-8
from vllm.v1.executor.multiproc_executor import MultiprocExecutor


class GCUMultiprocExecutor(MultiprocExecutor):
    uses_ray = False
    def _init_executor(self) -> None:
        import tops_extension.torch  # noqa: F401
        import torch_gcu  # noqa: F401
        import torch_gcu.transfer_to_gcu  # noqa: F401

        import vllm_gcu.compilation  # noqa: F401
        import vllm_gcu.distributed  # noqa: F401
        import vllm_gcu.kernels  # noqa: F401
        import vllm_gcu.patch  # noqa: F401

        return super()._init_executor()
