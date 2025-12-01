# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import threading
import time

from typing import TYPE_CHECKING, Any, Optional

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
    """
    Pop completed xfers by checking for DONE state.
    Args:
        transfers: dict of req_id -> list[running_xfer]
    Returns:
        set of req_ids that have all done xfers
    """
    done_req_ids: set[str] = set()
    for req_id, handles in list(transfers.items()):
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
    self.remote_agent_meta: Optional[NixlAgentMetadata] = None

def nixl_handshake(
    self,
    host: str,
    port: int,
    remote_tp_size: int,
    expected_engine_id: str,
) -> dict[int, str]:
    """Do a NIXL handshake with a remote instance."""

    start_time = time.perf_counter()

    # NOTE(rob): we need each rank to have a unique port. This is
    # a hack to keep us moving. We will switch when moving to etcd
    # or where we have a single ZMQ socket in the scheduler.

    # Handshake only with the remote TP rank that current local rank will
    # pull from. With homogeneous TP it happens to be the same rank_i.
    tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
    p_remote_rank = self.tp_rank // tp_ratio
    path = make_zmq_path("tcp", host, port + p_remote_rank)
    logger.debug("Querying metadata on path: %s at remote rank %s", path,
                    p_remote_rank)

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
        self.remote_agent_meta = metadata
        remote_agent_name = self.add_remote_agent(metadata, p_remote_rank,
                                                    remote_tp_size)
        setup_agent_time = time.perf_counter()
        logger.debug("NIXL handshake: add agent took: %s",
                        setup_agent_time - got_metadata_time)

    # Remote rank -> agent name.
    return {p_remote_rank: remote_agent_name}

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

    # Number of D TP workers reading from a single P TP worker. This is
    # 1 when P and D `--tensor-parallel-size` match.
    tp_ratio = divide(self._tp_size[self.engine_id],
                        self._tp_size[engine_id])
    assert tp_ratio > 0, "Decode TP cannot be smaller than prefill TP"
    assert not self._use_pallas or tp_ratio == 1, \
            "TPU (pallas_v1) DOES NOT support heterogeneous TP yet."

    # Handle tp_size>num_kv_heads: replicate KV cache.
    total_num_kv_heads = self.model_config.get_total_num_kv_heads()
    is_kv_replicated = self._tp_size[engine_id] // total_num_kv_heads >= 1

    remote_block_len = nixl_agent_meta.block_lens[0]
    if self.use_mla or is_kv_replicated:
        # With replicated KV cache, only the number of blocks can differ.
        assert self.block_len_per_layer == nixl_agent_meta.block_lens, \
            "KV cache sizes must match between P and D when replicated"
        remote_block_size = remote_block_len // (
            self.slot_size_per_layer[0])
    else:
        # When MLA is not used, this is a list of the same block length
        for block_len in nixl_agent_meta.block_lens:
            assert block_len == remote_block_len, \
                "All remote layers must have the same block size"
        remote_block_size = remote_block_len // (
            self.slot_size_per_layer[0] * tp_ratio)
        if self._use_flashinfer:
            # With flashinfer, KV are sent in the same message.
            remote_block_size //= 2
        if tp_ratio > 1:
            # Heterogeneous TP expects same kv_cache_layout.
            assert nixl_agent_meta.kv_cache_layout == self.kv_cache_layout
            if self.device_type == "xpu":
                raise ValueError(
                    "Heterogeneous TP is not supported on XPU")

        assert remote_block_len == self.block_len_per_layer[0] * tp_ratio, (
            "Remote P worker KV layer cache must be of shape [2, N, "
            "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
        )

    assert self.block_size == remote_block_size, (
        "Remote P worker with different page/block size is not supported "
        f"{self.block_size=}, {remote_block_size=}")

    # Create dst descs and xfer side handles. TP workers have same #blocks.
    if engine_id in self.dst_num_blocks:
        assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks
    else:
        self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

    # blocks_data = []
    # With homogeneous TP, D pulls the whole kv cache from corresponding
    # rank. With heterogeneous TP, prepare the descriptors by splitting the
    # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
    # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].
    self.kv_caches_base_addr[
        engine_id] = nixl_agent_meta.kv_caches_base_addr
    assert len(nixl_agent_meta.kv_caches_base_addr) == len(
        self.block_len_per_layer)

    return remote_agent_name



def read_blocks(self, local_block_ids: list[int],
                        remote_block_ids: list[int], dst_engine_id: str,
                        request_id: str):

        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # Number of D TP workers that will read from dst P. Propagate tp_ratio
        # on notification so that dst worker can wait before freeing blocks.
        tp_ratio = self._tp_size[
            self.engine_id] // self._tp_size[dst_engine_id]
        notif_id = f"{request_id}:{tp_ratio}".encode()
        remote_rank = self.tp_rank // tp_ratio
        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Get side handles.
        # local_xfer_side_handle = self.src_xfer_side_handle
        # remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs
        src_addrs = []
        dst_addrs = []
        layer_local_block_ids: list[int]
        layer_remote_block_ids: list[int]
        if not self.block_window_per_layer:
            # Default case: assume global attention
            layer_local_block_ids = local_block_ids
            layer_remote_block_ids = remote_block_ids
        else:
            # TODO(mgoin): remove this once we have hybrid memory allocator
            # Optimization for models with local attention (Llama 4)
            raise Exception("GCU: kv block merge transfer "
            "does not support block_window_per_layer yet")

        total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        is_kv_replicated = (
            self._tp_size[dst_engine_id] // total_num_kv_heads >= 1
        )
        # group by indices
        #contiguous blocks are grouped together
        if tp_ratio == 1 or self._use_mla:
            logger.debug(f"VLLM_GCU: group_concurrent_contiguous"
                        "is used for tp_ratio == 1 or use_mla")
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
        if self.remote_agent_meta is None:
            raise RuntimeError(
                f"remote_agent_meta is None when trying to read blocks from "
                f"engine {dst_engine_id}. Handshake may not have completed yet."
            )
        for layer_id in range(len(self.kv_caches_base_addr[self.engine_id])):
            src_ptr = self.kv_caches_base_addr[self.engine_id][layer_id]
            dst_ptr = self.kv_caches_base_addr[dst_engine_id][layer_id]
            # address offset
            src_item_len = self.block_len_per_layer[layer_id]
            # address offset
            dst_item_len = self.remote_agent_meta.block_lens[layer_id]
            # block length
            kv_block_len = (
                self.get_backend_aware_kv_block_len(layer_idx=layer_id)
            )
            rank_offset = (
                self.tp_rank % tp_ratio * kv_block_len
                if not (self.use_mla or is_kv_replicated) else 0
            )
            for prefill_index, decode_index in zip(
                    prefill_kv_blocks, dst_kv_blocks):
                length = kv_block_len * len(prefill_index)
                src_addr = (
                    src_ptr + int(prefill_index[0]) * src_item_len
                )
                dst_addr = (
                    dst_ptr
                    + int(decode_index[0]) * dst_item_len
                    + rank_offset
                )
                src_addrs.append((src_addr, length, self.tp_rank))
                dst_addrs.append((dst_addr, length, remote_rank))

                if self._use_flashinfer:
                    raise Exception(
                        "GCU: kv block merge transfer "
                        "does not support use_flashinfer yet"
                    )
                    # src_v_addr = src_addr + length
                    # dst_v_addr = (
                    #     dst_addr
                    #     + self.remote_agent_meta.block_lens[layer_id] // 2
                    # )
                    # src_addrs.append((src_v_addr, length, self.tp_rank))
                    # dst_addrs.append((dst_v_addr, length, remote_rank))

        src_descs = self.nixl_wrapper.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.nixl_wrapper.get_xfer_descs(dst_addrs, "VRAM")
        agent_name = self._remote_agents[dst_engine_id][remote_rank]
        # Prepare transfer with Nixl.
        handle = self.nixl_wrapper.initialize_xfer(
            "READ",
            src_descs,  #local descs  decode_descs
            dst_descs,  #remote descs   prefill_descs
            agent_name,  #remote_agent_name
            notif_id,  # type: ignore
        )
        if not handle:
            raise Exception("GCU: creating transfer failed.")

        # Begin async xfer.
        self.nixl_wrapper.transfer(handle)

        # Use handle to check completion in future step().
        self._recving_transfers[request_id].append(
            (handle, time.perf_counter()))


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
    if not envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorWorker._read_blocks",
            read_blocks).start()
