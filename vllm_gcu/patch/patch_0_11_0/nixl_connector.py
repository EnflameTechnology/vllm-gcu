# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import threading
import time

from typing import TYPE_CHECKING, Any, Optional, Tuple, List

import msgspec
import numpy as np
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.utils import divide
from vllm.logger import init_logger
from vllm.utils import make_zmq_path


from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnectorScheduler, NixlConnectorWorker
from vllm_gcu.utils import get_tx_mark_func
import vllm_gcu.envs as gcu_envs
import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlAgentMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import zmq_ctx
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import GET_META_MSG
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import EngineId

from unittest.mock import patch
import orjson

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None

try:
    from nixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None
    logger.warning("NIXL agent config is not available")


origin_get_num_new_matched_tokens = NixlConnectorScheduler.get_num_new_matched_tokens


def get_num_new_matched_tokens(self, request: "Request",
                               num_computed_tokens: int) -> tuple[int, bool]:
    ret = origin_get_num_new_matched_tokens(self, request, num_computed_tokens)
    params = request.kv_transfer_params
    if params is not None and params.get("do_remote_prefill") and \
            gcu_envs.VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE:
        logger.debug("VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE is enabled,"
                     "skipping first token")
        first_token = params.get("first_token", None)
        if first_token is not None:
            logger.debug(
                "NIXLConnector: first_token(%s) from kv_transfer_params",
                first_token)
            request.prompt_token_ids.append(first_token)
            request.num_prompt_tokens = len(request.prompt_token_ids)
            request._all_token_ids.append(first_token)
        else:
            logger.debug("NIXLConnector: no first_token in kv_transfer_params")
    return ret


origin_request_finished = NixlConnectorScheduler.request_finished


def request_finished(
    self,
    request: "Request",
    block_ids: list[int],
) -> tuple[bool, Optional[dict[str, Any]]]:
    async_save, txfer_params = origin_request_finished(self, request, block_ids)
    if txfer_params is None:
        return async_save, txfer_params
    if gcu_envs.VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE:
        logger.debug("VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE is enabled")
        # Get the first token from the request's output tokens
        first_token = None
        if request.num_output_tokens > 0:
            first_token = request.output_token_ids[0]
        else:
            logger.debug("No output tokens for request %s", request.request_id)
        txfer_params['first_token'] = first_token
    return async_save, txfer_params

def _pop_done_transfers(
        self, transfers: dict[str, list[tuple[int, float]]]) -> set[str]:
    done_req_ids: set[str] = set()
    
    # Check if we're in D_tp < P_tp mode (inverse_tp_ratio > 1)
    is_d_lt_p = getattr(self, 'inverse_tp_ratio', 1) > 1
    
    for req_id, handles in list(transfers.items()):
        if is_d_lt_p:
            # D_tp < P_tp: Two-pass approach - only release when ALL handles are done
            # First pass: check states without releasing
            all_done = True
            has_failed = False
            failed_state = None
            
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    pass  # Good, this one is done
                elif xfer_state == "PROC":
                    all_done = False
                else:
                    has_failed = True
                    failed_state = xfer_state
                    break
            
            if has_failed:
                raise RuntimeError("Transfer failed with state %s", failed_state)
            
            # Second pass: only release handles when ALL are done
            if all_done:
                for handle, _xfer_stime in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
                    self.xfer_stats.record_transfer()
                done_req_ids.add(req_id)
                del transfers[req_id]
        else:
            # D_tp >= P_tp: Original single-pass approach
            in_progress = False
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    self.nixl_wrapper.release_xfer_handle(handle)
                    # TODO (NickLucche) Get from NIXL telemetry once integrated
                    self.xfer_stats.record_transfer()
                elif xfer_state == "PROC":
                    in_progress = True
                    continue
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                        xfer_state)
            if not in_progress:
                if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
                    message = "_pop_done_transfers"
                    color = "blue"
                    domain = "VLLM"
                    category = "KVConnector"
                    payload = {
                        "req_ids": [req_id]
                    }
                    payload_str = orjson.dumps(payload)

                    tx_mark_func = get_tx_mark_func()
                    tx_mark_func(message, color, domain, category, payload_str)

                done_req_ids.add(req_id)
                del transfers[req_id]
            
    return done_req_ids

origin__init__ = NixlConnectorWorker.__init__

def __init__(self, vllm_config: VllmConfig, engine_id: str):
    origin__init__(self, vllm_config, engine_id)
    logger.info("Initializing GCU NIXL Connector Worker with Merge Block Transfer")

    # Heterogeneous TP support: D_tp >= P_tp or D_tp < P_tp
    self.inverse_tp_ratio: int = 1  # P_tp / D_tp when D_tp < P_tp

    # Store metadata and addresses per remote rank
    # Key: (engine_id, remote_rank) -> metadata
    # Used for both D_tp >= P_tp and D_tp < P_tp scenarios
    self._remote_agent_metas: dict[tuple[EngineId, int], NixlAgentMetadata] = {}
    # Key: (engine_id, remote_rank) -> list of base addresses per layer
    self._remote_kv_caches_base_addr: dict[tuple[EngineId, int], list[int]] = {}

def nixl_handshake(
    self,
    host: str,
    port: int,
    remote_tp_size: int,
    expected_engine_id: str,
) -> dict[int, str]:
    """Do a NIXL handshake with remote instance(s).
    
    Supports both D_tp >= P_tp and D_tp < P_tp scenarios:
    - D_tp >= P_tp: Each D worker connects to one P worker
    - D_tp < P_tp: Each D worker connects to multiple P workers
    """

    start_time = time.perf_counter()
    local_tp_size = self._tp_size[self.engine_id]
    result_agents: dict[int, str] = {}

    # NOTE(rob): we need each rank to have a unique port. This is
    # a hack to keep us moving. We will switch when moving to etcd
    # or where we have a single ZMQ socket in the scheduler.

    if local_tp_size >= remote_tp_size:
        logger.info(f"D_tp >= P_tp mode: D_worker_{self.tp_rank} will connect to "
                    f"P_worker {self.tp_rank // (local_tp_size // remote_tp_size)}")
        # D_tp >= P_tp: Each D worker connects to one P worker
        if local_tp_size % remote_tp_size != 0:
            raise ValueError(
                f"Heterogeneous TP (D >= P) requires D_tp_size ({local_tp_size}) "
                f"to be divisible by P_tp_size ({remote_tp_size}). "
                f"Got remainder: {local_tp_size % remote_tp_size}"
            )
        tp_ratio = local_tp_size // remote_tp_size
        p_remote_rank = self.tp_rank // tp_ratio
        remote_ranks_to_connect = [p_remote_rank]
    else:
        logger.info(
            f"D_tp < P_tp mode: D_worker_{self.tp_rank} will connect to "
            f"P_workers {list(range(
                self.tp_rank * (remote_tp_size // local_tp_size), 
                self.tp_rank * (remote_tp_size // local_tp_size) + (remote_tp_size // local_tp_size)
            ))}"
        )
        # D_tp < P_tp: Each D worker connects to multiple P workers
        if remote_tp_size % local_tp_size != 0:
            raise ValueError(
                f"Heterogeneous TP (D_tp < P_tp) requires P_tp_size ({remote_tp_size}) "
                f"to be divisible by D_tp_size ({local_tp_size}). "
                f"Got remainder: {remote_tp_size % local_tp_size}"
            )
        inverse_tp_ratio = remote_tp_size // local_tp_size
        base_remote_rank = self.tp_rank * inverse_tp_ratio
        remote_ranks_to_connect = list(range(
            base_remote_rank, 
            base_remote_rank + inverse_tp_ratio
        ))
        logger.info(f"D_tp < P_tp mode: D_worker_{self.tp_rank} will connect to "
                    f"P_workers {remote_ranks_to_connect}")

    for p_remote_rank in remote_ranks_to_connect:
        path = make_zmq_path("tcp", host, port + p_remote_rank)
        logger.debug("Querying metadata on path: %s at remote rank %s",
                    path, p_remote_rank)

        # Send query for the request.
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.debug("NIXL handshake: get metadata took: %s",
                        got_metadata_time - start_time)

            # Ensure engine id matches.
            if metadata.engine_id != expected_engine_id:
                raise RuntimeError(f"Remote NIXL agent engine ID mismatch. "
                                    f"Expected {expected_engine_id},"
                                    f"received {metadata.engine_id}.")

            # Register Remote agent.
            remote_agent_name = self.add_remote_agent(metadata, p_remote_rank,
                                                        remote_tp_size)
            result_agents[p_remote_rank] = remote_agent_name
            
            setup_agent_time = time.perf_counter()
            logger.debug("NIXL handshake: add agent for rank %s took: %s",
                        p_remote_rank, setup_agent_time - got_metadata_time)

    # Store the first remote rank for backward compatibility
    self.remote_tp_rank = remote_ranks_to_connect[0]
    
    # Remote rank -> agent name.
    return result_agents

def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Register the KV Cache data in nixl."""

    if self.use_host_buffer:
        self.initialize_host_xfer_buffer(kv_caches=kv_caches)
        assert len(self.host_xfer_buffers) == len(kv_caches), (
            f"host_buffer: {len(self.host_xfer_buffers)}, "
            f"kv_caches: {len(kv_caches)}")
        xfer_buffers = self.host_xfer_buffers
    else:
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, (
            "host_xfer_buffer should not be initialized when "
            f"kv_buffer_device is {self.kv_buffer_device}")

    logger.info(
        "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
        "use_host_buffer: %s", self.use_mla, self.kv_buffer_device,
        self.use_host_buffer)

    caches_data = []
    # With hybrid allocator, layers can share a kv cache tensor
    seen_base_addresses = []

    # Note(tms): I modified this from the original region setup code.
    # K and V are now in different regions. Advantage is that we can
    # elegantly support MLA and any cases where the K and V tensors
    # are non-contiguous (it's not locally guaranteed that they will be)
    # Disadvantage is that the encoded NixlAgentMetadata is now larger
    # (roughly 8KB vs 5KB).
    # Conversely for FlashInfer, K and V are registered in the same region
    # to better exploit the memory layout (ie num_blocks is the first dim).
    split_k_and_v = not (self.use_mla or self._use_pallas
                            or self._use_flashinfer)
    tensor_size_bytes = None
    # Enable different block lengths for different layers when MLA is used.
    self.block_len_per_layer = list[int]()
    self.slot_size_per_layer = list[int]()  # HD bytes in kv terms
    for layer_name, cache_or_caches in xfer_buffers.items():
        cache_list = cache_or_caches if split_k_and_v else [
            cache_or_caches
        ]

        for cache in cache_list:
            base_addr = cache.data_ptr()
            if base_addr in seen_base_addresses:
                continue

            seen_base_addresses.append(base_addr)
            curr_tensor_size_bytes = cache.numel() * cache.element_size()

            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes
                self.num_blocks = cache.shape[0]

            assert cache.shape[0] == self.num_blocks, \
                "All kv cache tensors must have the same number of blocks"

            self.block_len_per_layer.append(curr_tensor_size_bytes //
                                            self.num_blocks)
            self.slot_size_per_layer.append(self.block_len_per_layer[-1] //
                                            self.block_size)

            if not self.use_mla:
                # Different kv cache shape is not supported by HeteroTP
                assert tensor_size_bytes == curr_tensor_size_bytes, \
                    "All kv cache tensors must have the same size"
            caches_data.append(
                (base_addr, curr_tensor_size_bytes, self.tp_rank, ""))

    logger.debug("Different block lengths collected: %s",
                    set(self.block_len_per_layer))
    assert len(self.block_len_per_layer) == len(seen_base_addresses)
    assert self.num_blocks != 0

    self.kv_caches_base_addr[self.engine_id] = seen_base_addresses
    self.num_regions = len(caches_data)
    self.num_layers = len(xfer_buffers.keys())

    descs = self.nixl_wrapper.get_reg_descs(caches_data,
                                            self.nixl_memory_type)
    logger.debug("Registering descs: %s", caches_data)
    self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
    logger.debug("Done registering descs")
    self._registered_descs.append(descs)

    self.device_kv_caches = kv_caches
    self.dst_num_blocks[self.engine_id] = self.num_blocks
    if self._use_flashinfer:
        for i in range(len(self.slot_size_per_layer)):
            assert self.slot_size_per_layer[i] % 2 == 0
            self.slot_size_per_layer[i] //= 2

        # NOTE (NickLucche) When FlashInfer is used, memory is registered
        # with joint KV for each block. This minimizes the overhead in
        # registerMem allowing faster descs queries. In order to be able to
        # split on kv_heads dim as required by heterogeneous TP, one must
        # be able to index K/V separately. Hence we double the number
        # of 'virtual' regions here and halve `block_len` below.
        self.num_regions *= 2
    # TODO(mgoin): Hybrid memory allocator is currently disabled for
    # models with local attention (Llama 4). Can remove this once enabled.
    if self.vllm_config.model_config.hf_config.model_type == "llama4":
        from transformers import Llama4TextConfig
        assert isinstance(self.vllm_config.model_config.hf_text_config,
                            Llama4TextConfig)
        llama4_config = self.vllm_config.model_config.hf_text_config
        no_rope_layers = llama4_config.no_rope_layers
        chunk_size = llama4_config.attention_chunk_size
        chunk_block_size = math.ceil(chunk_size / self.block_size)
        for layer_idx in range(self.num_layers):
            # no_rope_layers[layer_idx] == 0 means NoPE (global)
            # Any other value means RoPE (local chunked)
            is_local_attention = no_rope_layers[layer_idx] != 0
            block_window = chunk_block_size if is_local_attention else None
            self.block_window_per_layer.append(block_window)
        logger.debug("Llama 4 block window per layer mapping: %s",
                        self.block_window_per_layer)
        assert len(self.block_window_per_layer) == self.num_layers

    # After KV Caches registered, listen for new connections.
    metadata = NixlAgentMetadata(
        engine_id=self.engine_id,
        agent_metadata=self.nixl_wrapper.get_agent_metadata(),
        kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
        num_blocks=self.num_blocks,
        block_lens=self.block_len_per_layer,
        attn_backend_name=self.backend_name,
        kv_cache_layout=self.kv_cache_layout)
    ready_event = threading.Event()
    self._nixl_handshake_listener_t = threading.Thread(
        target=self._nixl_handshake_listener,
        args=(metadata, ready_event, self.side_channel_port, self.tp_rank),
        daemon=True,
        name="nixl_handshake_listener")
    self._nixl_handshake_listener_t.start()
    ready_event.wait()  # Wait for listener ZMQ socket to be ready.

def add_remote_agent(self,
                         nixl_agent_meta: NixlAgentMetadata,
                         remote_tp_rank: int = 0,
                         remote_tp_size: int = 1) -> str:
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]

        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        else:
            assert self._tp_size[engine_id] == remote_tp_size
        # TODO We may eventually want to skip enforcing the same attn backend.
        assert nixl_agent_meta.attn_backend_name == self.backend_name

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata)

        # Calculate TP ratio between D and P
        local_tp_size = self._tp_size[self.engine_id]
        remote_tp_size_val = self._tp_size[engine_id]

        # Support both D_tp >= P_tp and D_tp < P_tp scenarios
        # Ensure TP sizes are divisible for correct data distribution
        if local_tp_size >= remote_tp_size_val:
            # D_tp >= P_tp: multiple D workers read from one P worker
            if local_tp_size % remote_tp_size_val != 0:
                raise ValueError(
                    f"Heterogeneous TP (D_tp >= P_tp) requires D_tp_size ({local_tp_size}) "
                    f"to be divisible by P_tp_size ({remote_tp_size_val})"
                )
            tp_ratio = local_tp_size // remote_tp_size_val
            inverse_tp_ratio = 1
        else:
            # D_tp < P_tp: one D worker reads from multiple P workers
            # D_tp < P_tp scenario only supports MLA architecture
            if not self.use_mla:
                raise ValueError(
                    f"Heterogeneous TP with D_tp < P_tp (D_tp={local_tp_size}, P_tp={remote_tp_size_val}) "
                    f"only supports MLA architecture. Current model does not use MLA."
                )
            if remote_tp_size_val % local_tp_size != 0:
                raise ValueError(
                    f"Heterogeneous TP (D_tp < P_tp) requires P_tp_size ({remote_tp_size_val}) "
                    f"to be divisible by D_tp_size ({local_tp_size})"
                )
            tp_ratio = 1
            inverse_tp_ratio = remote_tp_size_val // local_tp_size
            
        # Store for later use
        self.tp_ratio = tp_ratio
        self.inverse_tp_ratio = inverse_tp_ratio
        
        assert not self._use_pallas or (tp_ratio == 1 and inverse_tp_ratio == 1), \
               "TPU (pallas_v1) DOES NOT support heterogeneous TP yet."

        # Handle tp_size>num_kv_heads: replicate KV cache.
        total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        is_kv_replicated = self._tp_size[engine_id] // total_num_kv_heads >= 1

        remote_block_len = nixl_agent_meta.block_lens[0]
        # Derive the slot size on the remote side. For heterogeneous TP, the
        # local slot size is scaled by tp_ratio / inverse_tp_ratio relative to
        # the remote. We explicitly undo/redo that scaling to recover the
        # remote slot size for the upcoming block_size check.
        remote_slot_size = self.slot_size_per_layer[0]
        
        # For MLA or KV replicated cases, the KV cache is identical across all
        # TP workers, so we should NOT adjust the slot size based on TP ratio.
        # Only adjust for non-MLA, non-replicated cases where heads are split.
        if not (self.use_mla or is_kv_replicated):
            if tp_ratio > 1:
                remote_slot_size *= tp_ratio
            elif inverse_tp_ratio > 1:
                if remote_slot_size % inverse_tp_ratio != 0:
                    raise ValueError(
                        f"D_tp < P_tp: local slot_size {remote_slot_size} is not divisible by "
                        f"inverse_tp_ratio {inverse_tp_ratio}")
                remote_slot_size //= inverse_tp_ratio
                
        if remote_block_len % remote_slot_size != 0:
            raise ValueError(
                f"Remote block len {remote_block_len} not divisible by remote slot size "
                f"{remote_slot_size}")

        if self.use_mla or is_kv_replicated:
            # With replicated KV cache, only the number of blocks can differ.
            assert self.block_len_per_layer == nixl_agent_meta.block_lens, \
                "KV cache sizes must match between P and D when replicated"
            remote_block_size = remote_block_len // remote_slot_size
        else:
            # When MLA is not used, this is a list of the same block length
            for block_len in nixl_agent_meta.block_lens:
                assert block_len == remote_block_len, \
                    "All remote layers must have the same block size"
            
            # Block size on remote is derived using the remote slot size
            remote_block_size = remote_block_len // remote_slot_size
                
            if self._use_flashinfer:
                # With flashinfer, KV are sent in the same message.
                remote_block_size //= 2
                
            if tp_ratio > 1 or inverse_tp_ratio > 1:
                # Heterogeneous TP expects same kv_cache_layout.
                assert nixl_agent_meta.kv_cache_layout == self.kv_cache_layout
                if self.device_type == "xpu":
                    raise ValueError(
                        "Heterogeneous TP is not supported on XPU")

            if tp_ratio > 1:
                assert remote_block_len == self.block_len_per_layer[0] * tp_ratio, (
                    "Remote P worker KV layer cache must be of shape [2, N, "
                    "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
                )
            elif inverse_tp_ratio > 1:
                assert self.block_len_per_layer[0] == remote_block_len * inverse_tp_ratio, (
                    f"D_tp < P_tp: Local D worker KV layer cache must be {inverse_tp_ratio}x "
                    f"the size of remote P worker. Local: {self.block_len_per_layer[0]}, "
                    f"Remote: {remote_block_len}"
                )

        assert self.block_size == remote_block_size, (
            "Remote P worker with different page/block size is not supported "
            f"self.block_size={self.block_size}, remote_block_size={remote_block_size}")

        # Create dst descs and xfer side handles. TP workers have same #blocks.
        if engine_id in self.dst_num_blocks:
            # For D_tp < P_tp scenario, different P workers may report different num_blocks
            # in some edge cases. Use the minimum value to ensure safe access.
            if self.dst_num_blocks[engine_id] != nixl_agent_meta.num_blocks:
                if local_tp_size < remote_tp_size_val:
                    logger.warning(
                        f"D_tp < P_tp: P_worker_{remote_tp_rank} reports num_blocks="
                        f"{nixl_agent_meta.num_blocks}, but previously stored "
                        f"{self.dst_num_blocks[engine_id]}. Using minimum value."
                    )
                    self.dst_num_blocks[engine_id] = min(
                        self.dst_num_blocks[engine_id], 
                        nixl_agent_meta.num_blocks
                    )
                else:
                    assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks, (
                        f"P workers report different num_blocks: stored="
                        f"{self.dst_num_blocks[engine_id]}, received="
                        f"{nixl_agent_meta.num_blocks}"
                    )
        else:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        # Store KV cache base addresses and metadata
        # Key: (engine_id, remote_rank) for both D_tp >= P_tp and D_tp < P_tp
        self._remote_agent_metas[
            (engine_id, remote_tp_rank)] = nixl_agent_meta
        
        if local_tp_size < remote_tp_size_val:
            # D_tp < P_tp: store per remote rank to avoid overwriting
            self._remote_kv_caches_base_addr[
                (engine_id, remote_tp_rank)] = nixl_agent_meta.kv_caches_base_addr
            logger.debug(f"D_tp < P_tp: Stored KV cache addresses for P_worker_{remote_tp_rank}")
        else:
            # D_tp >= P_tp: original behavior
            self.kv_caches_base_addr[
                engine_id] = nixl_agent_meta.kv_caches_base_addr
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(
            self.block_len_per_layer)

        return remote_agent_name


def _read_blocks_d_ge_p(
    self, dst_engine_id: str, request_id: str, notif_id: bytes,
    prefill_kv_blocks: list, dst_kv_blocks: list,
    tp_ratio: int, is_kv_replicated: bool
):
    """Handle D_tp >= P_tp case: read partial heads from one P worker."""
    src_addrs = []
    dst_addrs = []
    
    # Get the remote rank for this D worker
    remote_rank = self.tp_rank // tp_ratio
    remote_meta = self._remote_agent_metas[(dst_engine_id, remote_rank)]
    
    for layer_id in range(len(self.kv_caches_base_addr[self.engine_id])):
        src_ptr = self.kv_caches_base_addr[self.engine_id][layer_id]
        dst_ptr = self.kv_caches_base_addr[dst_engine_id][layer_id]
        src_item_len = self.block_len_per_layer[layer_id]
        dst_item_len = remote_meta.block_lens[layer_id]
        kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=layer_id)
        
        # Calculate offset for this D worker's portion of P's KV cache
        rank_offset = self.tp_rank % tp_ratio * kv_block_len \
            if not (self.use_mla or is_kv_replicated) else 0
        
        for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
            length = kv_block_len * len(prefill_index)
            src_addr = src_ptr + int(prefill_index[0]) * src_item_len
            dst_addr = dst_ptr + int(decode_index[0]) * dst_item_len + rank_offset
            
            src_addrs.append((src_addr, length, self.tp_rank))
            dst_addrs.append((dst_addr, length, remote_rank))

            if self._use_flashinfer:
                raise Exception("NIXL connector not supported use_flashinfer")

    src_descs = self.nixl_wrapper.get_xfer_descs(src_addrs, self.nixl_memory_type)
    dst_descs = self.nixl_wrapper.get_xfer_descs(dst_addrs, self.nixl_memory_type)
    agent_name = self._remote_agents[dst_engine_id][remote_rank]
    
    handle = self.nixl_wrapper.initialize_xfer(
        "READ",
        src_descs,
        dst_descs,
        agent_name,
        notif_id,
    )
    if not handle:
        raise Exception("Creating transfer failed.")

    self.nixl_wrapper.transfer(handle)
    self._recving_transfers[request_id].append(
        (handle, time.perf_counter()))

def _read_blocks_d_lt_p_replicated(
    self, dst_engine_id: str, request_id: str, notif_id: bytes,
    prefill_kv_blocks: list, dst_kv_blocks: list,
    inverse_tp_ratio: int
):
    """
    Handle D_tp < P_tp case (MLA only).
    
    Note: D_tp < P_tp scenario only supports MLA architecture. This is enforced
    in add_remote_agent().
    
    For MLA, all P workers have identical KV cache content. We only need 
    to read from ONE P worker instead of all inverse_tp_ratio P workers.
    
    However, we still need to notify ALL P workers so they can release
    their blocks.
    
    Example with D=2, P=4, MLA=True:
    - D_worker_0 reads from P_worker_0 only (not P_worker_0 AND P_worker_1)
    - D_worker_0 notifies both P_worker_0 and P_worker_1
    - D_worker_1 reads from P_worker_2 only (not P_worker_2 AND P_worker_3)
    - D_worker_1 notifies both P_worker_2 and P_worker_3
    
    This saves (inverse_tp_ratio - 1) redundant reads per D worker.
    """
    base_remote_rank = self.tp_rank * inverse_tp_ratio
    
    # Only read from the first P worker in the range
    read_remote_rank = base_remote_rank
    
    # Check if this P worker is registered
    if read_remote_rank not in self._remote_agents.get(dst_engine_id, {}):
        raise RuntimeError(
            f"P worker rank {read_remote_rank} not registered. "
            f"Available: {list(self._remote_agents.get(dst_engine_id, {}).keys())}"
        )
    
    # Get the correct KV cache addresses for this specific P worker
    remote_key = (dst_engine_id, read_remote_rank)
    if remote_key not in self._remote_kv_caches_base_addr:
        raise RuntimeError(
            f"KV cache addresses for P_worker_{read_remote_rank} not found. "
            f"Available keys: {list(self._remote_kv_caches_base_addr.keys())}"
        )
    remote_kv_addrs = self._remote_kv_caches_base_addr[remote_key]
    remote_meta = self._remote_agent_metas[remote_key]
    
    src_addrs = []
    dst_addrs = []
    
    for layer_id in range(len(self.kv_caches_base_addr[self.engine_id])):
        # Local D worker's KV cache address
        src_ptr = self.kv_caches_base_addr[self.engine_id][layer_id]
        # This specific P worker's KV cache address
        dst_ptr = remote_kv_addrs[layer_id]
        
        src_item_len = self.block_len_per_layer[layer_id]
        dst_item_len = remote_meta.block_lens[layer_id]
        
        # For MLA/replicated, read full block (no offset needed)
        remote_kv_block_len = dst_item_len
        
        for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
            length = remote_kv_block_len * len(prefill_index)
            # Local destination: no offset for MLA/replicated
            src_addr = src_ptr + int(prefill_index[0]) * src_item_len
            # Remote source: full block from P worker
            dst_addr = dst_ptr + int(decode_index[0]) * dst_item_len
            
            src_addrs.append((src_addr, length, self.tp_rank))
            dst_addrs.append((dst_addr, length, read_remote_rank))

            if self._use_flashinfer:
                raise Exception("NIXL connector not supported use_flashinfer in D_tp < P_tp mode")

    src_descs = self.nixl_wrapper.get_xfer_descs(src_addrs, self.nixl_memory_type)
    dst_descs = self.nixl_wrapper.get_xfer_descs(dst_addrs, self.nixl_memory_type)
    agent_name = self._remote_agents[dst_engine_id][read_remote_rank]

    # Notification: only one D worker reads from this P worker
    p_notif_id = f"{request_id}:1".encode()
    
    handle = self.nixl_wrapper.initialize_xfer(
        "READ",
        src_descs,
        dst_descs,
        agent_name,
        p_notif_id,
    )
    if not handle:
        raise Exception(f"Creating transfer to P_worker_{read_remote_rank} failed.")

    self.nixl_wrapper.transfer(handle)
    self._recving_transfers[request_id].append(
        (handle, time.perf_counter()))
    
    logger.debug(
        f"D_tp < P_tp (MLA/replicated optimized) transfer initiated: "
        f"D_worker_{self.tp_rank} <- P_worker_{read_remote_rank}"
    )
    
    # Notify OTHER P workers (without reading) so they can release blocks
    # These P workers have identical data, we just need to let them know
    # the transfer is "complete" from their perspective
    for p_offset in range(1, inverse_tp_ratio):
        other_remote_rank = base_remote_rank + p_offset
        if other_remote_rank in self._remote_agents.get(dst_engine_id, {}):
            other_agent_name = self._remote_agents[dst_engine_id][other_remote_rank]
            self.nixl_wrapper.send_notif(other_agent_name, notif_msg=p_notif_id)
            logger.debug(
                f"D_tp < P_tp (MLA/replicated) notified P_worker_{other_remote_rank} "
                f"without reading (data is identical)"
            )
        else:
            logger.warning(
                "D_tp < P_tp: Cannot notify P_worker_%s for request %s: agent not registered. "
                "This may cause memory leak on the P worker.",
                other_remote_rank, request_id
            )

def read_blocks(self, local_block_ids: list[int],
                    remote_block_ids: list[int], dst_engine_id: str,
                    request_id: str):
    """
    Read KV cache blocks from remote P worker(s).
    
    Supports both D >= P and D_tp < P_tp scenarios:
    - D_tp >= P_tp: Read partial heads from one P worker
    - D_tp < P_tp: Read full heads from multiple P workers and merge
    """
    local_tp_size = self._tp_size[self.engine_id]
    remote_tp_size = self._tp_size[dst_engine_id]

    # Calculate tp_ratio for both directions
    # Ensure TP sizes are divisible for correct data distribution
    if local_tp_size >= remote_tp_size:
        if local_tp_size % remote_tp_size != 0:
            raise ValueError(
                f"Heterogeneous TP (D_tp >= P_tp) requires D_tp_size ({local_tp_size}) "
                f"to be divisible by P_tp_size ({remote_tp_size})"
            )
        tp_ratio = local_tp_size // remote_tp_size
        inverse_tp_ratio = 1
    else:
        if remote_tp_size % local_tp_size != 0:
            raise ValueError(
                f"Heterogeneous TP (D_tp < P_tp) requires P_tp_size ({remote_tp_size}) "
                f"to be divisible by D_tp_size ({local_tp_size})"
            )
        tp_ratio = 1
        inverse_tp_ratio = remote_tp_size // local_tp_size

    # Number of notifications P worker should expect
    # D_tp >= P_tp: tp_ratio D workers notify one P worker
    # D_tp < P_tp: one D worker notifies inverse_tp_ratio P workers (each gets 1 notif)
    notif_tp_ratio = tp_ratio if local_tp_size >= remote_tp_size else 1
    notif_id = f"{request_id}:{notif_tp_ratio}".encode()

    # Full prefix cache hit: do not need to read remote blocks,
    # just notify P worker(s) that we have the blocks we need.
    num_local_blocks = len(local_block_ids)
    if num_local_blocks == 0:
        if local_tp_size >= remote_tp_size:
            # D_tp >= P_tp: notify one P worker
            remote_rank = self.tp_rank // tp_ratio
            if remote_rank not in self._remote_agents.get(dst_engine_id, {}):
                logger.warning(
                    "Cannot notify P_worker_%s for request %s: agent not registered",
                    remote_rank, request_id
                )
                return
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
        else:
            # D_tp < P_tp: notify all P workers this D worker is connected to
            base_remote_rank = self.tp_rank * inverse_tp_ratio
            for offset in range(inverse_tp_ratio):
                remote_rank = base_remote_rank + offset
                if remote_rank not in self._remote_agents.get(dst_engine_id, {}):
                    logger.warning(
                        "Cannot notify P_worker_%s for request %s: agent not registered",
                        remote_rank, request_id
                    )
                    continue
                agent_name = self._remote_agents[dst_engine_id][remote_rank]
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
        return

    # Partial prefix cache hit: just read uncomputed blocks.
    num_remote_blocks = len(remote_block_ids)
    assert num_local_blocks <= num_remote_blocks
    if num_local_blocks < num_remote_blocks:
        remote_block_ids = remote_block_ids[-num_local_blocks:]

    if not self.block_window_per_layer:
        layer_local_block_ids = local_block_ids
        layer_remote_block_ids = remote_block_ids
    else:
        raise Exception("NIXL connector not supported block_window_per_layer")

    total_num_kv_heads = self.model_config.get_total_num_kv_heads()
    # Note: use `>` instead of `>=` to correctly handle the borderline case
    # where tp_size == num_kv_heads (each worker has exactly 1 unique head, not replicated)
    is_kv_replicated = self._tp_size[dst_engine_id] > total_num_kv_heads

    # Group contiguous blocks for efficient transfer
    should_group = (
        self.use_mla or
        (local_tp_size == remote_tp_size)
    )
    if should_group:
        logger.debug(f"VLLM_GCU: group_concurrent_contiguous"
                        "is used for D_tp == P_tp or use_mla")
        if len(layer_local_block_ids) == 0:
                prefill_kv_blocks = []
                dst_kv_blocks = []
        else:
            src_arr = np.array(layer_local_block_ids, dtype=np.int32)
            dst_arr = np.array(layer_remote_block_ids, dtype=np.int32)
            # Find break points where either src or dst indices
            # are not consecutive
            brk = np.where((np.diff(src_arr) != 1) |
                        (np.diff(dst_arr) != 1))[0] + 1
            src_groups = np.split(src_arr, brk)
            dst_groups = np.split(dst_arr, brk)

            # Convert back to lists
            prefill_kv_blocks = [g.tolist() for g in src_groups]
            dst_kv_blocks = [g.tolist() for g in dst_groups]
    else:
        prefill_kv_blocks = [[x] for x in layer_local_block_ids]
        dst_kv_blocks = [[x] for x in layer_remote_block_ids]

    # Check if handshake has completed by verifying metadata exists
    if not self._remote_agent_metas:
        raise RuntimeError(
            f"No remote agent metadata available when trying to read blocks from "
            f"engine {dst_engine_id}. Handshake may not have completed yet."
        )

    if local_tp_size >= remote_tp_size:
        # D_tp >= P_tp: Original logic - read from one P worker
        self._read_blocks_d_ge_p(
            dst_engine_id, request_id, notif_id,
            prefill_kv_blocks, dst_kv_blocks,
            tp_ratio, is_kv_replicated
        )
    else:
        # D_tp < P_tp: Only MLA is supported (checked in add_remote_agent)
        # MLA: all P workers have identical KV cache, only need to read from one
        logger.debug(
            "D_tp < P_tp with MLA: optimized single P worker read"
        )
        self._read_blocks_d_lt_p_replicated(
            dst_engine_id, request_id, notif_id,
            prefill_kv_blocks, dst_kv_blocks,
            inverse_tp_ratio
        )


if envs.VLLM_NVTX_SCOPES_FOR_PROFILING and (not gcu_envs.VLLM_GCU_ENABLE_NIXL_BLOCK_MERGE_TRANSFER):
    origin_read_blocks = NixlConnectorWorker._read_blocks
if envs.VLLM_NVTX_SCOPES_FOR_PROFILING and gcu_envs.VLLM_GCU_ENABLE_NIXL_BLOCK_MERGE_TRANSFER:
    origin_read_blocks = read_blocks

def _read_blocks(self, local_block_ids: list[int],
                    remote_block_ids: list[int], dst_engine_id: str,
                    request_id: str):
    origin_read_blocks(self, local_block_ids, remote_block_ids, dst_engine_id, request_id)

    message = "_read_blocks"
    color = "blue"
    domain = "VLLM"
    category = "KVConnector"
    payload = {
        "req_ids": [request_id]
    }
    payload_str = orjson.dumps(payload)

    tx_mark_func = get_tx_mark_func()
    tx_mark_func(message, color, domain, category, payload_str)


origin_shutdown = NixlConnectorWorker.shutdown
def shutdown(self):
    origin_shutdown(self)
    # Clean up D_tp < P_tp specific data structures
    self._remote_agent_metas.clear()
    self._remote_kv_caches_base_addr.clear()



patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorScheduler.get_num_new_matched_tokens",
    get_num_new_matched_tokens).start()
patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorScheduler.request_finished",
    request_finished).start()

patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker._pop_done_transfers",
    _pop_done_transfers).start()

if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
    patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker._read_blocks",
        _read_blocks).start()
if gcu_envs.VLLM_GCU_ENABLE_NIXL_BLOCK_MERGE_TRANSFER:
    logger.debug(f"VLLM_GCU: VLLM_GCU_ENABLE_NIXL_BLOCK_MERGE_TRANSFER is enabled")

    NixlConnectorWorker._read_blocks_d_ge_p = _read_blocks_d_ge_p
    NixlConnectorWorker._read_blocks_d_lt_p_replicated = _read_blocks_d_lt_p_replicated
    patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker.__init__",
        __init__).start()
    patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker._nixl_handshake",
        nixl_handshake).start()
    patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker.register_kv_caches",
        register_kv_caches).start()
    patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker.add_remote_agent",
        add_remote_agent).start()
    patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker.shutdown",
        shutdown).start()
    if not envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker._read_blocks",
            read_blocks).start()
