import torch
from contextlib import nullcontext
from typing import Callable, Optional, Any
from unittest.mock import patch
from torch.cuda import device
from vllm.utils import round_up

from vllm.config import VllmConfig, CUDAGraphMode
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

    num_tokens_across_dp = torch.tensor([num_tokens_per_ubatch],
                                        dtype=torch.int32)
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

    def __init__(self, runnable: Callable[..., Any], vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode, device: device):
        super().__init__(runnable, vllm_config, runtime_mode, device)
        self.query_start_loc_ubatch = [
            torch.empty((vllm_config.scheduler_config.max_num_seqs + 1, ),
                         device=device,
                         dtype=torch.int32) for i in range(2)
        ]

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

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:

            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE,
                                          CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = (ubatch_slices[0].token_slice.stop -
                      ubatch_slices[0].token_slice.start) * 2
        input_ids = kwargs['input_ids']
        positions = kwargs['positions']
        intermediate_tensors = kwargs['intermediate_tensors']
        inputs_embeds = kwargs['inputs_embeds']
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        # assert dp_metadata is not None
        ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE)

        for idx, meta in enumerate(ubatch_metadata):
            attn_metadata = meta.context.forward_context.attn_metadata
            if attn_metadata is not None:
                for layer_idx, name in enumerate(attn_metadata):
                    n = attn_metadata[name].query_start_loc.shape[0]
                    if layer_idx == 0:
                        self.query_start_loc_ubatch[idx][:n].copy_(attn_metadata[name].query_start_loc)
                    attn_metadata[name].query_start_loc = self.query_start_loc_ubatch[idx][:n]

        if num_tokens not in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:

            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.model)
        elif num_tokens in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            with self.sm_control:
                return self._run_ubatches(ubatch_metadata, self.model)


patch('vllm.v1.worker.gpu_ubatch_wrapper.UBatchWrapper',
      GCUUBatchWrapper).start()
patch('vllm.v1.worker.gpu_model_runner.UBatchWrapper',
      GCUUBatchWrapper).start()
patch('vllm.v1.worker.ubatch_splitting.get_dp_padding_ubatch',
      get_dp_padding_ubatch).start()
