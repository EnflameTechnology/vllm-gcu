#!/usr/bin/env python
# coding=utf-8
from typing import List, Tuple, Union

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.simple_connector import SimpleConnector
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import SimpleBuffer
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

import vllm_gcu.kernels._custom_ops as ops
from vllm_gcu.attention.backends.mla import GCUMLAMetadata
from vllm_gcu.attention.backends.xformers import GCUXFormersMetadata
from vllm_gcu.distributed.kv_transfer.pyeccl_pipe import PyEcclPipe


logger = init_logger(__name__)


class Connector(SimpleConnector):
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.config = config.kv_transfer_config
        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.tp_size = config.parallel_config.tensor_parallel_size

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer = None
        self.consumer_buffer = None

        port_offset_base = 2 * rank

        if self.config.is_kv_producer:
            self.producer_data_pipe = PyEcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base,
                device="gcu",
            )
            self.producer_signal_pipe = PyEcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base + 1,
                device="cpu",
            )

            self.producer_buffer = SimpleBuffer(
                self.producer_signal_pipe,
                self.producer_data_pipe,
                self.config.kv_buffer_size,
            )
        else:
            self.consumer_data_pipe = PyEcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base,
                device="gcu",
            )
            self.consumer_signal_pipe = PyEcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base + 1,
                device="cpu",
            )
            self.consumer_buffer = SimpleBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
            )

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        num_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                if isinstance(model_input.attn_metadata, GCUXFormersMetadata):

                    key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                    value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
                elif isinstance(model_input.attn_metadata, GCUMLAMetadata):
                    kv_cache = kv_cache.reshape(-1, num_heads, head_size)
                    key_cache = kv_cache[:, :, :512]
                    value_cache = kv_cache[:, :, 512:]

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)

            self.insert(
                current_tokens,
                torch.ones_like(current_tokens, dtype=bool),
                keys,
                values,
                hidden_or_intermediate_states[start_pos:end_pos],
            )

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors],
        bool,
        "ModelInputForGPUWithSamplingMetadata",
    ]:

        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # This can happen during inflight batching. See:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning(
                    "You should set --enable_chunked_prefill=False "
                    "and --max_num_batched_tokens "
                    "should be equal to max_seq_len_to_capture"
                )
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            ret = self.select(
                current_tokens, torch.ones_like(current_tokens, dtype=bool)
            )
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[1]
            keys: torch.Tensor = ret[2]
            values: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]

            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(
                model_executable.model.start_layer, model_executable.model.end_layer
            ):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                if isinstance(model_input.attn_metadata, GCUXFormersMetadata):
                    key_cache, value_cache = kv_cache[0], kv_cache[1]

                    x = 16 // kv_cache.element_size()
                    num_blocks = kv_cache.shape[1]

                    num_kv_heads = layer.self_attn.attn.impl.num_kv_heads
                    head_size = layer.self_attn.attn.impl.head_size
                    key_cache = key_cache.view(
                        num_blocks, num_kv_heads, head_size // x, -1, x
                    )
                    value_cache = value_cache.view(
                        num_blocks, num_kv_heads, head_size, -1
                    )
                    ops.reshape_and_cache(
                        keys[i - model_executable.model.start_layer].to(
                            key_cache.device
                        ),
                        values[i - model_executable.model.start_layer].to(
                            value_cache.device
                        ),
                        key_cache,
                        value_cache,
                        slot_mapping[start_pos:end_pos].flatten(),
                        layer.self_attn.attn.kv_cache_dtype,
                        layer.self_attn.attn._k_scale_float,
                        layer.self_attn.attn._v_scale_float,
                    )
                elif isinstance(model_input.attn_metadata, GCUMLAMetadata):
                    ops.concat_and_cache_mla(
                        keys[i - model_executable.model.start_layer]
                        .to(kv_cache.device)
                        .squeeze(1),
                        values[i - model_executable.model.start_layer]
                        .to(kv_cache.device)
                        .squeeze(1),
                        kv_cache,
                        slot_mapping[start_pos:end_pos],
                        layer.self_attn.mla_attn.kv_cache_dtype,
                        layer.self_attn.mla_attn._k_scale,
                    )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            logger.debug(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.",
                torch.distributed.get_rank(),
            )
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.",
                torch.distributed.get_rank(),
            )
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0
            )

        return hidden_or_intermediate_states, bypass_model_exec, model_input
