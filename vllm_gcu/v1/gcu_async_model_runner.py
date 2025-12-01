import torch
import os
import gc
import numpy as np
import asyncio
import time
import vllm.envs as envs
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional, Union, Any
from unittest.mock import patch
# import vllm.device_allocator
from vllm.utils import MemorySnapshot, GiB_bytes
from vllm.model_executor import set_random_seed
from vllm.distributed.parallel_state import get_pp_group

from vllm.v1.utils import report_usage_stats
from vllm_gcu import gcumem
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import round_up, supports_dynamo
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.block_table import BlockTable
from vllm_gcu.utils import set_gcu_forward_context
from vllm.forward_context import (DPMetadata, get_forward_context,
                                  set_forward_context)
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, graph_capture, is_global_first_rank,
    prepare_communication_buffer_for_model)
from tqdm import tqdm
from vllm.logger import init_logger
from vllm.compilation.counter import compilation_counter
from vllm_gcu.v1.worker.async_gcu_input_batch import AsyncGCUInputBatch
from vllm_gcu.worker.worker_v1 import GCUModelRunner
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.sampling_params import SamplingType
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

with patch("vllm.forward_context.set_forward_context", set_gcu_forward_context):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

@dataclass
class ExecuteMetaData:
    num_scheduled_tokens: int = 0
    num_input_tokens: int = 0
    attn_metadata: Any = None
    attention_cuda_graphs: bool = False
    num_scheduled_tokens_np: np.ndarray = None
    logits_indices: torch.Tensor = None
    num_tokens_across_dp: torch.Tensor = None
    spec_decode_metadata: SpecDecodeMetadata = None
    prepare_event: torch.cuda.Event = None
    exec_buffer: int = 0
    token_indices: torch.Tensor = None
    cu_num_tokens: np.ndarray = None
    def wait_prepare_finish(self):
        """
            等待prepare工作完成
        """
        assert self.prepare_event is not None, "prepare_event is None" 
        self.prepare_event.wait()



class AsyncGCUModelRunner(GCUModelRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_batch = AsyncGCUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
        )
        additional_config = self.vllm_config.additional_config
        
        async_executing  = additional_config.get("async_executing", False)

        assert async_executing, "only by enabling async_executing will enable AsyncGCUModelRunner!"
        
        # 使能的buffer_id
        self.exec_buffer = 0

        self.all_input_ids = torch.zeros(2, self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)

        self.input_ids = self.all_input_ids[self.exec_buffer]
        
        self.all_positions = torch.zeros(2, self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)

        self.positions = self.all_positions[self.exec_buffer]

        self.all_query_start_loc = torch.zeros(2, self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        
        self.query_start_loc = self.all_query_start_loc[self.exec_buffer]
        
        
        self.all_seq_lens = torch.zeros(2, self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        

        self.seq_lens = self.all_seq_lens[self.exec_buffer]
        
        
        self.all_slot_mapping = torch.zeros(2, self.max_num_tokens,
                                        dtype=torch.int64,
                                        device=self.device)

        self.slot_mapping = self.all_slot_mapping[self.exec_buffer]

        self.all_inputs_embeds = torch.zeros(
            (2, self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)
        
        self.inputs_embeds = self.all_inputs_embeds[self.exec_buffer]

        self.exec_id = 0


        self.req_ids = [[], []]
        
        self.req_id_to_index = [{}, {}]

        self.async_copy_stream = torch.cuda.Stream()

        self.token_indices_device = torch.empty(size=(2, self.max_num_tokens), device=self.device, dtype=torch.int32)

        self.input_ids_index_tensor_device = torch.empty(size=(2, self.max_num_tokens),
                                            device=self.device,
                                            dtype=torch.int32)
        
        self.prev_common_req_indices_tensor_device = torch.empty(size=(2, self.max_num_reqs),
                                        dtype=torch.int32,
                                        device=self.device)

        self.flattened_indices_cpu = torch.empty(size=(self.max_num_reqs, ), dtype=torch.int32, pin_memory=True)
        self.flattened_indices_np = self.flattened_indices_cpu.numpy()

        self.prev_common_req_indices_cpu = torch.empty(size=(self.max_num_reqs, ), dtype=torch.int32, pin_memory=True)
        self.prev_common_req_indices_np = self.prev_common_req_indices_cpu.numpy()

        self.token_indices_cpu = torch.empty(size=(self.max_num_tokens, ), dtype=torch.int32, pin_memory=True)
        self.token_indices_np = self.token_indices_cpu.numpy()

    def switch_engine(self, exec_buffer: Optional[int] = None):
        """
        Args:
            exec_buffer (int, optional): 切换到执行的CUDA Graph Buffer.
        """

        if exec_buffer is not None:
            assert exec_buffer in [0, 1], f"exec_buffer must be 0 or 1, but got {exec_buffer}"
            if self.exec_buffer == exec_buffer:
                return
            self.exec_buffer = exec_buffer
        else:
            self.exec_buffer = (self.exec_buffer + 1) % 2
        
        new_exec_buffer = self.exec_buffer
        
        self.input_ids = self.all_input_ids[new_exec_buffer]
        
        self.positions = self.all_positions[new_exec_buffer]

        self.query_start_loc = self.all_query_start_loc[new_exec_buffer]
        
        self.seq_lens = self.all_seq_lens[new_exec_buffer]
        
        self.slot_mapping = self.all_slot_mapping[new_exec_buffer]

        self.inputs_embeds = self.all_inputs_embeds[new_exec_buffer]

        self.input_batch.block_table.switch_buffer(exec_buffer)


    def _prepare_metadata(self, scheduler_output: "SchedulerOutput"):
        """
            准备模型推理需要的 attention metadata
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)
        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # print("self.requests", self.requests)
        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        self.token_indices_np[:total_num_scheduled_tokens] = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices = self.token_indices_cpu[:total_num_scheduled_tokens]
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[
                kv_cache_group_id]
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
            # where K is the max_num_blocks_per_req and the block size is 2.
            # NOTE(woosuk): We can't simply use `token_indices // block_size`
            # here because M (max_model_len) is not necessarily divisible by
            # block_size.
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                positions_np // block_size)
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        seq_len_max = self.seq_lens_cpu[:num_reqs].max().item()
        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc[num_reqs + 1:].fill_(
            self.query_start_loc_cpu[num_reqs].item())

        query_start_loc = self.query_start_loc[:num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
        )
        setattr(common_attn_metadata, 'max_query_len_item', seq_len_max)
        attn_metadata: dict[str, Any] = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):

            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            builder = self.attn_metadata_builders[kv_cache_group_id]
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(
                    num_scheduled_tokens,
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id],
                    kv_cache_group_spec.kv_cache_spec,
                    builder,
                )
            
            attn_metadata_i = (builder.build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
            ))

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        attention_cuda_graphs = all(
            b.can_run_in_cudagraph(common_attn_metadata)
            for b in self.attn_metadata_builders)

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            
            
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return attention_cuda_graphs, attn_metadata, logits_indices, spec_decode_metadata, num_scheduled_tokens, token_indices, cu_num_tokens

    def _prepare_input_tokens(self, scheduler_output: "SchedulerOutput", execute_meta_data: ExecuteMetaData = 0):
        """
            准备模型推理需要的input token
        """

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        token_indices = execute_meta_data.token_indices
        exec_buffer = execute_meta_data.exec_buffer

        token_indices_device = self.token_indices_device[exec_buffer, :len(token_indices)]

        # prepare计算出来的token_indices放到拷贝流上执行
        with self.set_async_copy_stream():
            token_indices_device.copy_(
                token_indices, non_blocking=False)

        if self.input_batch.prev_sampled_token_ids is None:

            torch.index_select(self.input_batch.token_ids_device_tensor.flatten(),
                            0,
                            token_indices_device, out=self.input_ids[:scheduler_output.total_num_scheduled_tokens])
            return
        
        cu_num_tokens = execute_meta_data.cu_num_tokens

        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        
        assert prev_req_id_to_index is not None
        
        flattened_indices = []
        
        prev_common_req_indices = []
        
        indices_match = True if not self.speculative_config else False
        
        max_flattened_index = -1
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                if not self.speculative_config:
                    flattened_index = cu_num_tokens[cur_index] - 1
                    flattened_indices.append(flattened_index)
                    indices_match &= (prev_index == flattened_index)
                    max_flattened_index = max(max_flattened_index, flattened_index)
                else:
                    # 推测解码还需支持
                    pass                    

        num_common_tokens = len(flattened_indices)
         
        if num_common_tokens < total_num_scheduled_tokens:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            # self.input_ids[:total_num_scheduled_tokens].copy_(
            #     self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
            torch.index_select(self.input_batch.token_ids_device_tensor.flatten(),
                            0,
                            token_indices_device, out=self.input_ids[:scheduler_output.total_num_scheduled_tokens])

        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids_cpu will have all the input ids.
            return

        if indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens,
                                                        0],
                non_blocking=True)
            return

        # 在拷贝流上执行index相关的拷贝
        with self.set_async_copy_stream():

            input_ids_index_tensor = self.input_ids_index_tensor_device[exec_buffer, :num_common_tokens]
                
            self.flattened_indices_np[:len(flattened_indices)] = np.array(flattened_indices)

            input_ids_index_tensor.copy_(
                    self.flattened_indices_cpu[:len(flattened_indices)], non_blocking=True)

            prev_common_req_indices_tensor = self.prev_common_req_indices_tensor_device[exec_buffer, :len(prev_common_req_indices)]
        
            self.prev_common_req_indices_np[:len(prev_common_req_indices)] = np.array(prev_common_req_indices)
        
            prev_common_req_indices_tensor.copy_(
                self.prev_common_req_indices_cpu[:len(prev_common_req_indices)],
                non_blocking=True
            )

            input_ids_index_tensor_i64 = input_ids_index_tensor.to(dtype=torch.int64, non_blocking=False)

        # input_ids 必须在默认流上执行
        self.input_ids.scatter_(
            dim=0,
            index=input_ids_index_tensor_i64,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0])

    @torch.inference_mode()
    def _prepare_execute_meta(
        self,
        scheduler_output: "SchedulerOutput",
        exec_buffer=0
    ) -> ExecuteMetaData:
        # Part1：
        self._update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            return ExecuteMetaData(exec_buffer=exec_buffer)
        
        prepare_event = torch.cuda.Event()
        # 将当前的req_ids、req_id_to_index拷贝一份，防止input_batch中的被删除
        self.req_ids[exec_buffer] = self.input_batch.req_ids.copy()
        self.req_id_to_index[exec_buffer] = self.input_batch.req_id_to_index.copy()
        
        # Prepare the decoder inputs.
        attention_cuda_graphs, attn_metadata, logits_indices, \
        spec_decode_metadata, num_scheduled_tokens_np, token_indices, cu_num_tokens = self._prepare_metadata(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        prepare_event.record()
        execute_meta_data = ExecuteMetaData(num_scheduled_tokens=num_scheduled_tokens,
                                        num_input_tokens=num_input_tokens,
                                        attn_metadata=attn_metadata,
                                        attention_cuda_graphs=attention_cuda_graphs,
                                        num_scheduled_tokens_np=num_scheduled_tokens_np,
                                        logits_indices=logits_indices,
                                        num_tokens_across_dp=num_tokens_across_dp,
                                        spec_decode_metadata=spec_decode_metadata,
                                        prepare_event=prepare_event,
                                        exec_buffer=exec_buffer,
                                        token_indices=token_indices,
                                        cu_num_tokens=cu_num_tokens)

        return execute_meta_data

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        exec_buffer: int = 0
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:

        self.switch_engine(exec_buffer)
        with self.set_async_copy_stream():
            execute_meta_data = self._prepare_execute_meta(scheduler_output, exec_buffer)
        task = self._create_async_execute_task(scheduler_output, execute_meta_data)

        return task


    def get_dp_padding(self,
                       num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        """
            开启DP时，vllm会调用num_tokens_across_dp（cpu all_gather）获取每个Core Engine推理的token数量，
            原来这一步是在set_forward_context里执行的，为了能够跟Device重叠，我们将其移到get_dp_padding中。
        """
        dp_size = self.parallel_config.data_parallel_size
        dp_rank = self.parallel_config.data_parallel_rank
        num_tokens_across_dp = None
        if dp_size > 1:
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                            num_tokens, dp_size, dp_rank)
        return 0, num_tokens_across_dp

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        capture_attn_cudagraph: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata
        from vllm.v1.spec_decode.eagle import EagleProposer

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        if num_tokens > 0:
            num_tokens += num_pad

            # Set num_scheduled_tokens based on num_tokens and max_num_seqs
            # for dummy run with LoRA so that the num_reqs collectively
            # has num_tokens in total.
            assert num_tokens <= self.scheduler_config.max_num_batched_tokens
            max_num_reqs = self.scheduler_config.max_num_seqs
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
            assert sum(num_scheduled_tokens_list) == num_tokens
            assert len(num_scheduled_tokens_list) == num_reqs
            num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        else:
            num_reqs = 1
            num_scheduled_tokens = np.array([], dtype=np.int32)

        attn_metadata = None
        if capture_attn_cudagraph:
            attn_metadata = {}

            query_start_loc = self.query_start_loc[: num_reqs + 1]
            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(
                self.seq_lens_cpu[:num_reqs], non_blocking=True
            )
            seq_lens = self.seq_lens[:num_reqs]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=num_tokens,
            )

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):
                attn_metadata_i = self.attn_metadata_builders[
                    kv_cache_group_id
                ].build_for_cudagraph_capture(common_attn_metadata)
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False
                )

            with (
                self.maybe_randomize_inputs(input_ids),
                set_gcu_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    exec_buffer=self.exec_buffer
                ),
            ):
                outputs = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "set -O %s and ensure `use_cudagraph` was not manually set to "
                "False", CompilationLevel.PIECEWISE)
            return

        compilation_counter.num_gpu_runner_capture_triggers += 1
        
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.

        self.switch_engine(0)
        with graph_capture(device=self.device):
            full_cg = self.full_cuda_graph
            # Only rank 0 should #print progress bar during capture
            compilation_cases = reversed(self.cudagraph_batch_sizes)
            if is_global_first_rank():
                compilation_cases = tqdm(list(compilation_cases),
                                         desc=f"Capturing CUDA graph shapes({self.exec_buffer})")

            for num_tokens in compilation_cases:
                # We skip EPLB here since we don't want to record dummy metrics
                for _ in range(
                        self.compilation_config.cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens,
                                    capture_attn_cudagraph=full_cg,
                                    skip_eplb=True)
                self._dummy_run(num_tokens,
                                capture_attn_cudagraph=full_cg,
                                skip_eplb=True)

        self.switch_engine(1)

        with graph_capture(device=self.device):

            compilation_cases = reversed(self.cudagraph_batch_sizes)

            if is_global_first_rank():
                compilation_cases = tqdm(list(compilation_cases),
                                         desc=f"Capturing CUDA graph shapes({self.exec_buffer})")

            for num_tokens in compilation_cases:
                # 执行 n 次预热
                for _ in range(
                        self.compilation_config.cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens,
                                    capture_attn_cudagraph=full_cg,
                                    skip_eplb=True)
                # 开始捕获
                self._dummy_run(num_tokens,
                                capture_attn_cudagraph=full_cg,
                                skip_eplb=True)
                
        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    async def _create_async_execute_task(self, scheduler_output: SchedulerOutput, 
                             execute_meta_data: ExecuteMetaData) -> ModelRunnerOutput | Any:
        if not execute_meta_data.num_input_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)
        
        exec_buffer = execute_meta_data.exec_buffer
        
        num_scheduled_tokens = execute_meta_data.num_scheduled_tokens
        
        num_input_tokens = execute_meta_data.num_input_tokens
        
        spec_decode_metadata = execute_meta_data.spec_decode_metadata

        execute_finish_event = torch.cuda.Event()

        self.switch_engine(exec_buffer)
        self._prepare_input_tokens(scheduler_output, execute_meta_data)

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []
        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, but we
        # compiled with full CUDA graphs, we have to skip them entirely.
        skip_cuda_graphs = self.full_cuda_graph and not execute_meta_data.attention_cuda_graphs

        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        # Run the model.
        # Use persistent buffers for CUDA graphs.

        execute_meta_data.wait_prepare_finish()
        with set_gcu_forward_context(
                execute_meta_data.attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=execute_meta_data.num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
                exec_buffer=exec_buffer
        ):
            self.maybe_setup_kv_connector(scheduler_output)
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None


        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  execute_meta_data.num_scheduled_tokens_np, finished_sending,
                                  finished_recving)

            sample_hidden_states = hidden_states[execute_meta_data.logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)

        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = set()
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.add(i)
        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        
        max_gen_len = sampled_token_ids.shape[-1]
        
        if max_gen_len == 1:
            # No Spec decode tokens
            self.input_batch.prev_sampled_token_ids = \
                sampled_token_ids
        else:
            self.input_batch.prev_num_rejected_tokens_calc_event = None

        assert self.input_batch.prev_sampled_token_ids.shape[-1] == 1
        
        req_ids = self.req_ids[exec_buffer]

        self.input_batch.prev_sampled_token_ids_invalid_indices = \
                    discard_sampled_tokens_req_indices


        self.input_batch.prev_req_id_to_index = {
            req_id: i
            for i, req_id in enumerate(req_ids)
            if i not in discard_sampled_tokens_req_indices
        }

        for req_idx in range(sampled_token_ids.shape[0]):

            if req_idx in discard_sampled_tokens_req_indices:
                continue

            sampled_ids = sampled_token_ids[req_idx]
            
            req_id = req_ids[req_idx]
            
            if req_id in self.input_batch.req_id_to_index:
                req_idx = self.input_batch.req_id_to_index[req_id]

                start_idx = self.input_batch.num_tokens_no_spec[req_idx]

                end_idx = start_idx + sampled_ids.shape[0]

                assert end_idx <= self.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.max_model_len}")

                # self.input_batch.token_ids_device_tensor[req_idx,
                #                             start_idx:end_idx].copy_(sampled_token_ids[req_idx], non_blocking=True)

                self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                self.input_batch.num_tokens[req_idx] = end_idx
        # 判断Device端是否执行完毕，如果没有执行完
        # model_runner会放弃执行权限，交给Worker
        # 告诉处理output的协程，kernel已经launch完毕

        execute_finish_event.record()

        while not execute_finish_event.query():
            await asyncio.sleep(0)

        self.switch_engine(exec_buffer)
        with self.set_async_copy_stream():
            if max_gen_len == 1:
                # No spec decode tokens.
                t1 = time.time()
                valid_sampled_token_ids = sampled_token_ids.tolist()
                t2 = time.time()
                if t2 - t1 > 0.01:
                    print(f"{t2} tolist耗时异常: {t2 - t1}")
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):

            if not sampled_ids:
                continue

            req_id = req_ids[req_idx]

            if req_id in self.requests:
                req_state = self.requests[req_id]
                req_state.output_token_ids.extend(sampled_ids)

        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        else:
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                execute_meta_data.attn_metadata,
            )

        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()
        self.eplb_step()
        # 等待异步拷贝流的操作完成
        self.async_copy_stream.synchronize()
        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.req_id_to_index[exec_buffer],
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            num_nans_in_logits=num_nans_in_logits,
        )
        return model_runner_output

    @contextmanager
    def set_async_copy_stream(self):
        """
        设置当前异步拷贝流的上下文管理器，退出上下文后，将自动恢复之前的流
        
        参数:
            stream (torch.cuda.Stream): 要设置为当前流的 CUDA 流对象
        
        """
        # 保存当前流
        current_stream = torch.cuda.current_stream()
        try:
            # 设置新流
            torch.cuda.set_stream(self.async_copy_stream)
            yield  # 在此处执行用户代码
        finally:
            torch.cuda.set_stream(current_stream)

with patch("vllm.device_allocator", "cumem", gcumem):
    from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
