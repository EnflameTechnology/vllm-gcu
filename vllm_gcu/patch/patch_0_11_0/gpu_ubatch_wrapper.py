import torch
from contextlib import nullcontext
from typing import Callable, Optional
from unittest.mock import patch
from vllm.utils import round_up

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
import vllm.envs as envs
from vllm.utils import has_deep_gemm
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper, SMControlContextManager, UbatchMetadata
from vllm.forward_context import get_forward_context
from vllm.v1.worker.ubatch_splitting import get_dp_padding_ubatch, is_second_ubatch_empty

origin_get_dp_padding_ubatch = get_dp_padding_ubatch


def get_dp_padding_ubatch(
        num_tokens_unpadded: int, num_tokens_padded: int,
        should_attempt_ubatching: bool,
        vllm_config: VllmConfig) -> tuple[bool, Optional[torch.Tensor]]:
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size > 1:
        return origin_get_dp_padding_ubatch(num_tokens_unpadded,
                                            num_tokens_padded,
                                            should_attempt_ubatching,
                                            vllm_config)
    # If this DP rank doesn't want to attempt microbatching
    if not should_attempt_ubatching:
        num_tokens_across_dp = torch.tensor([0], dtype=torch.int32)
        return False, num_tokens_across_dp

    # Round up to the next multiple of two for even divisibility
    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = True

    if is_second_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        should_ubatch = False

    num_tokens_across_dp = torch.tensor([num_tokens_per_ubatch], dtype=torch.int32)
    return should_ubatch, num_tokens_across_dp


class SipControlContextManager(SMControlContextManager):

    def __init__(self, comm_sms: int, set_comm_sms: Callable[[int], None],
                 set_compute_sms: Callable[[int], None]):
        """
        Context manager for controlling SM (Streaming Multiprocessor) 
        allocation. Upon entering the context, it sets the number of SMs
        allocated for communication and computation to comm_sms and
        total_sms - comm_sms respectively. Upon exiting, it restores the
        allocation to use all available SMs (i.e. total_sms).

        Args:
            comm_sms (int): The number of SMs to allocate for communication. 
                (The remainder will be used for computation.)
            set_comm_sms (Callable[[int], None]): 
                A function that sets the number of SMs for communication.
            set_compute_sms (Callable[[int], None]): 
                A function that sets the number of SMs for computation.
        """

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # TODO: remove hardcode after torch fix
        total_sms = 24  #props.multi_processor_count

        assert comm_sms < total_sms
        self.total_sms = total_sms
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms


class GCUUBatchWrapper(UBatchWrapper):

    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        if envs.VLLM_ALL2ALL_BACKEND == 'deepep_low_latency':
            # deepep ll background comm does not need compute resource
            return nullcontext()
        if envs.VLLM_ALL2ALL_BACKEND != 'deepep_high_throughput':
            # alltoall runs on rudundant sip, others not supported DBO
            return nullcontext()
        comm_sms = envs.VLLM_DBO_COMM_SMS

        set_comm_sms = lambda sms: None
        if vllm_config.parallel_config.enable_expert_parallel:
            # Currently only DeepEP highthroughput supports SM control so this
            # only affects that case.
            all2all_manager = get_ep_group(
            ).device_communicator.all2all_manager

            if all2all_manager.max_sms_used() is not None:
                comm_sms = min(comm_sms, all2all_manager.max_sms_used())

            if comm_sms > 0:
                set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        if has_deep_gemm() and comm_sms > 0:
            import deep_gemm as dg
            set_compute_sms = lambda sms: dg.set_num_sms(sms)

        return SipControlContextManager(comm_sms=comm_sms,
                                        set_comm_sms=set_comm_sms,
                                        set_compute_sms=set_compute_sms)

    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        with patch('torch.cuda.current_blas_handle', lambda: None):
            return super()._capture_ubatches(ubatch_metadata, model)

    def _make_ubatch_metadata(self, *args, **kwargs) -> list[UbatchMetadata]:
        forward_context = get_forward_context()
        metas = super()._make_ubatch_metadata(*args, **kwargs)
        if hasattr(forward_context, 'all2allv_threshold'):
            for i in metas:
                i.context.forward_context.all2allv_threshold = forward_context.all2allv_threshold
        return metas


patch('vllm.v1.worker.gpu_ubatch_wrapper.UBatchWrapper',
      GCUUBatchWrapper).start()
patch('vllm.v1.worker.gpu_model_runner.UBatchWrapper',
      GCUUBatchWrapper).start()
patch('vllm.v1.worker.ubatch_splitting.get_dp_padding_ubatch',
      get_dp_padding_ubatch).start()
