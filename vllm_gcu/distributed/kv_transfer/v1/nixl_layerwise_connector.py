# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import httpx
import msgspec
import numpy as np
import torch
import zmq

from vllm import envs
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group, get_world_group)
from vllm.distributed.utils import divide
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend, current_platform
from vllm.utils import make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.utils import current_stream
import vllm_gcu.envs as gcu_envs

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

EngineId = str
ReqId = str
LayerName = str

GET_META_MSG = b"get_meta_msg"
DONE_SENDING_MSG = b"done_sending_msg"
FIRST_TOKEN_MSG = b"first_token_msg"

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
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

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": ("cuda", ),
    "tpu": ("cpu", ),
    "xpu": ("cpu", ),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())
_ENABLE_FIRST_TOKEN_REUSE = gcu_envs.VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE
_INVALID_TOKEN_ID: int = -1

_SCHEDULER_PORT_OFFSET: int = 0
_WORKER_BASE_PORT_OFFSET: int = 1024
_WORKER_RECV_PORT_OFFSET: int = 5120


class NixlAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_lens: list[int]
    attn_backend_name: str
    kv_cache_layout: str
    layer_names: list[LayerName] = field(default_factory=list)


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: Optional[str]
    remote_tp_size: int
    meta_server: Optional[str]


class LayerTransferMeta:
    """Per-layer transfer metadata for WRITE mode"""
    def __init__(self, request_id: ReqId, req_meta: ReqMeta, layer_index: int,
                 local_handle: int, remote_handle: int,
                 local_block_descs_ids: np.ndarray,
                 remote_block_descs_ids: np.ndarray,
                 expiration_time: float,
                 xfer_handler: Optional[int] = None,
                 event: Optional[torch.cuda.Event] = None):
        self.request_id: ReqId = request_id
        self.req_meta: ReqMeta = req_meta
        self.layer_index: int = layer_index
        self.local_handle: int = local_handle
        self.remote_handle: int = remote_handle
        self.local_block_descs_ids: np.ndarray = local_block_descs_ids
        self.remote_block_descs_ids: np.ndarray = remote_block_descs_ids
        self.xfer_handler: int = xfer_handler
        self.event: Optional[torch.cuda.Event] = event
        self.expiration_time: float = expiration_time


    def __hash__(self):
        return hash((self.request_id, self.layer_index))

    def __eq__(self, other):
        if not isinstance(other, LayerTransferMeta):
            return False
        return (
            self.request_id == other.request_id and
            self.layer_index == other.layer_index
        )

    def __repr__(self):
        return f"LayerTransferMeta(request_id={self.request_id}, " \
               f"layer_index={self.layer_index}, " \
               f"xfer_handler={self.xfer_handler})"


class KVCacheLayerwiseSendThread(threading.Thread):

    def __init__(self,
                 num_layers: int,
                 device: torch.device,
                 ready_event: threading.Event,
                 nixl_wrapper,
                 callback_func: Callable[..., None] = lambda x: None):
        super().__init__(daemon=True, name="KVCacheLayerwiseSendThread")
        self.num_layers = num_layers
        self.nixl_wrapper = nixl_wrapper

        self.req_status: dict[ReqId, list[int]] = defaultdict(list)

        self.lock = threading.Lock()
        self.send_queue = queue.Queue[LayerTransferMeta]()
        self.done_requests: set[str] = set()
        self.timeout_requests: set[str] = set()

        self.device = device
        self.ready_event = ready_event
        self.callback_func = callback_func


    def run(self):
        current_platform.set_device(self.device)
        logger.info("KVCacheLayerwiseSendThread run on %s", self.device)
        self.ready_event.set()
        while True:
            layer_transfer_meta = self.send_queue.get()
            request_id = layer_transfer_meta.request_id
            xfer_handler = layer_transfer_meta.xfer_handler
            if request_id in self.timeout_requests:
                if xfer_handler is not None:
                    self.nixl_wrapper.release_xfer_handle(xfer_handler)
                continue

            if xfer_handler is None:
                if layer_transfer_meta.event is not None:
                    if not layer_transfer_meta.event.query():
                        logger.debug("KVCacheLayerwiseSendThread event not "
                                     "set, trans_meta: %s",
                                     layer_transfer_meta)
                        now = time.perf_counter()
                        if now < layer_transfer_meta.expiration_time:
                            time.sleep(0.0005)
                            self.send_queue.put(layer_transfer_meta)
                        else:
                            self.timeout_requests.add(request_id)
                            logger.warning("Releasing expired KV blocks for "
                                "request %s which has completed %d layer(s) "
                                "within %d seconds.", request_id,
                                len(self.req_status[request_id]),
                                envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT)
                            with self.lock:
                                self.done_requests.add(request_id)
                                self.req_status.pop(request_id, None)
                        continue
                write_handle = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    layer_transfer_meta.local_handle,
                    layer_transfer_meta.local_block_descs_ids,
                    layer_transfer_meta.remote_handle,
                    layer_transfer_meta.remote_block_descs_ids,
                )
                self.nixl_wrapper.transfer(write_handle)
                layer_transfer_meta.xfer_handler = write_handle
                logger.debug("KVCacheLayerwiseSendThread to check_xfer_state, "
                             "write_handle: %s, trans_meta: %s",
                             write_handle, layer_transfer_meta)
                self.send_queue.put(layer_transfer_meta)
                continue

            xfer_state = self.nixl_wrapper.check_xfer_state(xfer_handler)
            logger.debug("KVCacheLayerwiseSendThread trans_meta: %s, "
                         "xfer_state: %s", layer_transfer_meta, xfer_state)
            if xfer_state == "DONE":
                self.nixl_wrapper.release_xfer_handle(xfer_handler)
                with self.lock:
                    self.req_status[request_id].append(
                        layer_transfer_meta.layer_index)
            elif xfer_state == "PROC":
                now = time.perf_counter()
                if now < layer_transfer_meta.expiration_time:
                    self.send_queue.put(layer_transfer_meta)
                else: # timeout
                    self.timeout_requests.add(request_id)
                    logger.warning(
                        "Releasing expired KV blocks for request %s which has "
                        "completed %d layer(s) within %d seconds.", request_id,
                        len(self.req_status[request_id]),
                        envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT)
                    self.nixl_wrapper.release_xfer_handle(xfer_handler)
                    # Notify P server to free blocks
                    with self.lock:
                        self.done_requests.add(request_id)
                        self.req_status.pop(request_id, None)
                continue
            else:
                raise RuntimeError("Transfer failed with state %s", xfer_state)

            if len(self.req_status[request_id]) == self.num_layers:
                self.callback_func(layer_transfer_meta)
                with self.lock:
                    self.done_requests.add(request_id)
                    self.req_status.pop(request_id, None)


    def add_transfer_meta(self, transfer_meta: LayerTransferMeta):
        self.send_queue.put(transfer_meta)


    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.lock:
            finished_requests = self.done_requests
            self.done_requests = set()
        return finished_requests


class KVCacheLayerwiseRecvThread(threading.Thread):

    def __init__(self,
                 recv_thread_port: int,
                 tp_size: int,
                 tp_rank: int,
                 ready_event: threading.Event):
        super().__init__(daemon=True, name="KVCacheLayerwiseRecvThread")
        self.recv_thread_port = recv_thread_port
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.ready_event = ready_event

        self.lock = threading.Lock()
        self.done_recv_requests = set[str]()
        self.done_first_token_requests = set[str]()
        self.task_tracker = dict[str, int]()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        port = self.recv_thread_port + self.tp_rank
        path = make_zmq_path("tcp", host, port)
        logger.info("KVCacheLayerwiseRecvThread "
                    "starting listening on path: %s", path)

        with zmq_ctx(zmq.ROUTER, path) as sock:
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    msg = decoder.decode(payload[0])
                    if msg[0] == DONE_SENDING_MSG:
                        notif_msg = msg[1]
                        req_id, tp_ratio = notif_msg.rsplit(":", 1)
                        logger.debug("KVCacheLayerwiseRecvThread get "
                                     "DONE_RECVING_MSG for "
                                     "request: %s, tp_ratio: %s",
                                     req_id, tp_ratio)
                        self.update_recv_task(req_id, int(tp_ratio))
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == FIRST_TOKEN_MSG:
                        notif_msg = msg[1]
                        req_id, first_token = notif_msg.rsplit(":", 1)
                        logger.debug("KVCacheLayerwiseRecvThread get "
                                     "FIRST_TOKEN_MSG for "
                                     "request: %s, first_token: %s",
                                     req_id, first_token)
                        self.update_first_token_task(req_id, int(first_token))
                        sock.send_multipart((identity, b"", b"ACK"))
                    else:
                        logger.error("Connection listener got "
                                     "unexpected message %s", msg)
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)

    def update_recv_task(self, req_id, tp_ratio):
        with self.lock:
            self.task_tracker[req_id] += 1
            if self.task_tracker[req_id] == tp_ratio:
                self.task_tracker.pop(req_id)
                self.done_recv_requests.add(req_id)

    def update_first_token_task(self, req_id, first_token):
        with self.lock:
            self.done_first_token_requests.add(req_id)

    def add_task_trace(self, req_id):
        with self.lock:
            self.task_tracker[req_id] = 0

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        if _ENABLE_FIRST_TOKEN_REUSE and self.tp_rank == 0:
            finished_requests = set[str]()
            with self.lock:
                done_first_token_reqs = self.done_first_token_requests.copy()
                for req_id in done_first_token_reqs:
                    if req_id in self.done_recv_requests:
                        finished_requests.add(req_id)
                        self.done_recv_requests.remove(req_id)
                        self.done_first_token_requests.remove(req_id)
        else:
            with self.lock:
                finished_requests = self.done_recv_requests
                self.done_recv_requests = set()
        return finished_requests


class NixlLayerwiseConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[ReqId, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ):
        """Add a new request to transfer metadata with layerwise support"""

        _req = ReqMeta(
            local_block_ids=local_block_ids,
            # D workers don't need to receive these from proxy here.
            remote_tp_size=kv_transfer_params.get("remote_tp_size", 1),
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id", None),
            remote_host=kv_transfer_params.get("remote_host", None),
            remote_port=kv_transfer_params.get("remote_port", None),

            meta_server=kv_transfer_params.get("meta_server", None),
        )

        self.requests[request_id] = _req



class NixlLayerwiseConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = NixlLayerwiseConnectorMetadata()
        self._connector_metadata_mtp = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = \
                NixlLayerwiseConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlLayerwiseConnectorWorker(
                vllm_config, self.engine_id)

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            logger.warning_once("Unable to detect current VLLM config. "
                                "Fallback to default kv cache layout.")
            return None
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # return None when we have mla
            # as the layout should not matter in that case,
            # which fallback to the default behavior.
            return None
        logger.info_once("NixlLayerwiseConnector setting KV cache "
                         "layout to HND for better xfer performance.")
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[Optional[int], bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_kv_connector_stats(self) -> Optional[KVConnectorStats]:
        assert self.connector_worker is not None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
            cls,
            data: Optional[dict[str,
                                Any]] = None) -> Optional[KVConnectorStats]:
        return NixlKVConnectorStats(data=data) if data is not None \
            else NixlKVConnectorStats()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        if self._connector_metadata is None:
            logger.warning("_connector_metadata is none in start_load_kv")
            return
        assert isinstance(self._connector_metadata,
                          NixlLayerwiseConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for a specific layer to complete loading"""
        assert self.connector_worker is not None
        if self._connector_metadata is None:
            logger.debug("wait_for_layer_load for mtp, "
                         "layer_name: %s", layer_name)
            self._connector_metadata = self._connector_metadata_mtp
        assert isinstance(self._connector_metadata,
                          NixlLayerwiseConnectorMetadata)
        self.connector_worker.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                     attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Layerwise WRITE mode: initiate transfer for specific layer"""
        assert self.connector_worker is not None
        if self._connector_metadata is None:
            logger.debug("save_kv_layer for mtp, layer_name: %s", layer_name)
            self._connector_metadata = self._connector_metadata_mtp
        assert isinstance(self._connector_metadata,
                          NixlLayerwiseConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer,
                                            attn_metadata,
                                            self._connector_metadata,
                                            **kwargs)
        self._connector_metadata_mtp = self._connector_metadata

    def wait_for_save(self):
        """NixlLayerwiseConnector does not save explicitly."""
        pass

    def shutdown(self):
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        if self.connector_worker is not None:
            self.connector_worker.shutdown()


class NixlLayerwiseConnectorScheduler:
    """Implementation of Scheduler side methods with layerwise support"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            _SCHEDULER_PORT_OFFSET +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)
        logger.info("Initializing NIXL Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}

        # req_id, (len(prompt), local_block_ids, request)
        self._reqs_need_send_layerwise: \
            dict[ReqId, tuple[int, list[int], Request]] = {}

        # first token reuse
        self._reqs_in_process: dict[ReqId, Request] = {}
        # Background thread for handling first token
        self._first_token_listener_t: Optional[threading.Thread] = None
        # Protects _reqs_in_process.
        self._first_token_lock = threading.RLock()

        if _ENABLE_FIRST_TOKEN_REUSE and \
            self.vllm_config.kv_transfer_config.is_kv_consumer:
            ready_event = threading.Event()
            self._first_token_listener_t = threading.Thread(
                target=self._first_token_listener_on_decode,
                args=(ready_event, self.side_channel_host,
                      self.side_channel_port),
                daemon=True,
                name="first_token_listener")
            self._first_token_listener_t.start()
            ready_event.wait()


    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "NixlLayerwiseConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        # for decode
        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        params = request.kv_transfer_params
        logger.debug(
            "NixlLayerwiseConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if not params:
            return

        # for decode
        if params.get("do_remote_prefill"):
            local_block_ids = (blocks.get_unhashed_block_ids()
                               if num_external_tokens > 0 else [])
            # Get unhashed blocks to pull from remote.
            self._reqs_need_recv[request.request_id] = (
                request,
                local_block_ids)

            params["do_remote_prefill"] = False

        # Layerwise prefiller add request need send
        if params.get("do_remote_decode"):
            local_block_ids = (blocks.get_block_ids()[0])
            self._reqs_need_send_layerwise[request.request_id] = (len(
                request.all_token_ids), local_block_ids, request)


    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlLayerwiseConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None

            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
            if _ENABLE_FIRST_TOKEN_REUSE:
                with self._first_token_lock:
                    self._reqs_in_process[req_id] = req

        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        for req_id, new_blocks in zip(cached_reqs.req_ids,
                                      cached_reqs.new_block_ids):
            if req_id in self._reqs_need_send_layerwise \
                and new_blocks is not None:
                total_tokens, block_ids, req = \
                    self._reqs_need_send_layerwise[req_id]
                logger.debug("Chunked_prefill, before extern computed_tokens, "
                             "req_id: %s, total_tokens: %s, block_ids: %s",
                             req_id, total_tokens, block_ids)
                block_ids.extend(new_blocks[0])
                logger.debug("Chunked_prefill, after extern computed_tokens, "
                             "req_id: %s, total_tokens: %s, block_ids: %s",
                             req_id, total_tokens, block_ids)

        computed_tokens = dict(
            list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens)) +
            [(x.req_id, x.num_computed_tokens) for x in new_reqs])
        for req_id, scheduled_tokens in \
            scheduler_output.num_scheduled_tokens.items():
            if req_id in self._reqs_need_send_layerwise:
                total_tokens, block_ids, req = \
                    self._reqs_need_send_layerwise[req_id]
                computed_token = computed_tokens.get(req_id, 0)
                current_tokens = computed_token + scheduled_tokens
                logger.debug("Chunked_prefill, with scheduled_tokens, "
                             "req_id: %s, total_tokens: %s, "
                             "current_tokens: %s, block_ids: %s",
                             req_id, total_tokens, current_tokens, block_ids)
                # current_tokens may greater than total_tokens when enable mtp
                if current_tokens >= total_tokens:
                    meta.add_new_req(
                        request_id=req_id,
                        local_block_ids=block_ids,
                        kv_transfer_params=req.kv_transfer_params,
                    )
                    if _ENABLE_FIRST_TOKEN_REUSE:
                        with self._first_token_lock:
                            self._reqs_in_process[req_id] = req
                    self._reqs_need_send_layerwise.pop(req_id)


        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NixlLayerwiseConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)
        if not params:
            return False, None


        if (not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        if _ENABLE_FIRST_TOKEN_REUSE and params.get("do_remote_decode"):
            first_token = _INVALID_TOKEN_ID
            if request.num_output_tokens > 0:
                first_token = request.output_token_ids[0]
            else:
                logger.debug("No output tokens for request %s",
                             request.request_id)

            remote_host = request.kv_transfer_params.get("remote_host", None),
            remote_port = request.kv_transfer_params.get("remote_port", None),
            assert remote_host is not None and remote_port is not None, \
                f"Invalid remote host({remote_host}) " \
                f"or remote port({remote_port})"
            self._send_first_token_to_decode_sched(request.request_id,
                                                   first_token,
                                                   remote_host,
                                                   remote_port)

        delay_free_blocks = len(block_ids) > 0

        return delay_free_blocks, None


    def _send_first_token_to_decode_sched(self, req_id: ReqId, first_token: int,
                                          remote_host: Union[str, tuple],
                                          remote_port: Union[int, tuple]):
        try:
            if isinstance(remote_host, tuple):
                if len(remote_host) == 0:
                    raise ValueError(f"remote_host is empty tuple "
                                     f"for request {req_id}")
                remote_host = remote_host[0]
            if isinstance(remote_port, tuple):
                if len(remote_port) == 0:
                    raise ValueError(f"remote_port is empty tuple "
                                     f"for request {req_id}")
                remote_port = remote_port[0]
            # get scheduler port from worker port, using offset
            remote_sched_port = (remote_port - _WORKER_BASE_PORT_OFFSET +
                                 _SCHEDULER_PORT_OFFSET)
            logger.debug("Sending first token(%s) to decode sched for "
                         "request %s to %s:%d",
                         first_token, req_id, remote_host, remote_sched_port)
            path = make_zmq_path("tcp", remote_host, remote_sched_port)
            send_msg = f"{req_id}:{first_token}:{remote_port}"
            with zmq_ctx(zmq.REQ, path) as sock:
                msg_encoder = msgspec.msgpack.Encoder()
                encoded_data = msg_encoder.encode((FIRST_TOKEN_MSG, send_msg))
                sock.send(encoded_data)
                ack = sock.recv()
                if ack != b"ACK":
                    raise ValueError(f"Unexpected ACK response: {ack}")
                logger.debug("ConnectorScheduler send_first_token_to_decode "
                             "for req: %s", req_id)

        except Exception as e:
            logger.error(
                f"Sending first token for request {req_id} to "
                f"{remote_host}:{remote_sched_port} fail with error: {e}"
            )


    def _first_token_listener_on_decode(self, ready_event: threading.Event,
                                        host: str, port: int):
        """
        Background thread for recv first token from
        P NixlLayerwiseConnectorScheduler.
        """
        path = make_zmq_path("tcp", host, port)
        logger.debug("first_token_listener starting listening "
                     "on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    msg = decoder.decode(payload[0])
                    if msg[0] == FIRST_TOKEN_MSG:
                        recv_msg = msg[1]
                        req_id, first_token_str, worker_port_str = \
                            recv_msg.rsplit(":", 2)
                        logger.debug("_first_token_listener get FIRST_TOKEN_MSG"
                                     " for request: %s, first_token: %s",
                                     req_id, first_token_str)
                        self._set_first_token_to_request(
                            req_id, int(first_token_str))
                        self._send_first_token_msg_to_worker(
                            req_id, int(first_token_str), int(worker_port_str))
                        sock.send_multipart((identity, b"", b"ACK"))
                    else:
                        logger.error("first_token listener got "
                                     "unexpected message %s", msg)
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)


    def _set_first_token_to_request(self, req_id: ReqId, first_token: int):
        with self._first_token_lock:
            assert req_id in self._reqs_in_process, \
                f"Invalid request {req_id} for setting first token."
            request = self._reqs_in_process[req_id]
            if first_token != _INVALID_TOKEN_ID:
                logger.debug("Reuse first_token[%s] for request %s",
                             first_token, req_id)
                request.prompt_token_ids.append(first_token)
                request.num_prompt_tokens = len(request.prompt_token_ids)
                request._all_token_ids.append(first_token)
            else:
                logger.debug("Reuse first_token get _INVALID_TOKEN_ID, "
                             "request: %s", req_id)
            self._reqs_in_process.pop(req_id)


    def _send_first_token_msg_to_worker(self, req_id: ReqId, first_token: int,
                                        worker_port: int):
        try:
            host = self.side_channel_host
            recv_port_offset = _WORKER_RECV_PORT_OFFSET - \
                               _WORKER_BASE_PORT_OFFSET
            port = worker_port + recv_port_offset
            logger.debug("Sending first token msg to worker "
                         "for request %s to %s:%d",
                         req_id, host, port)
            path = make_zmq_path("tcp", host, port)
            notif_msg = f"{req_id}:{first_token}"
            with zmq_ctx(zmq.REQ, path) as sock:
                msg_encoder = msgspec.msgpack.Encoder()
                encoded_data = msg_encoder.encode((FIRST_TOKEN_MSG, notif_msg))
                sock.send(encoded_data)
                ack = sock.recv()
                if ack != b"ACK":
                    raise ValueError(f"Unexpected ACK response: {ack}")
                logger.debug("NixlLayerwiseConnectorScheduler, Sending first "
                             "token msg to worker for request: %s", req_id)

        except Exception as e:
            logger.error("Sending first token msg to worker for request %s "
                         "to %s:%s fail with error: %s", req_id,
                         host, port)


    def shutdown(self):
        """Shutdown the connector scheduler."""
        if self.self._first_token_listener_t is not None:
            self.self._first_token_listener_t.join(timeout=0)
            self.self._first_token_listener_t = None

        logger.info("NixlLayerwiseConnectorScheduler shutdown complete")

class NixlLayerwiseConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        self.nixl_backends = \
            vllm_config.kv_transfer_config.get_from_extra_config(
                "backends", ["UCX"])
        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        if nixl_agent_config is None:
            config = None
        else:
            config = nixl_agent_config(backends=self.nixl_backends) if len(
                non_ucx_backends) > 0 else nixl_agent_config(num_threads=8)

        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)
        # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

        # NIXL handshake port.
        # NOTE(rob): Within a DP group, each DP rank gets its own
        # base port (which is sent in the KVTransferParams).
        # Each TP rank listens/queries on the base_port + tp_rank.
        self.side_channel_port: int = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            _WORKER_BASE_PORT_OFFSET +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST


        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = \
            vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[
                self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported.")
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        self.nixl_memory_type = current_platform.get_nixl_memory_type()
        if self.nixl_memory_type is None:
            if self.kv_buffer_device == "cuda":
                self.nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                self.nixl_memory_type = "DRAM"
        if self.nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported.")

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        self.total_layer_names: list[LayerName] = []

        # nixl_prepped_dlist_handle.
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[EngineId, int] = {}

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: Optional[threading.Thread] = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator")
        self._ready_requests = queue.Queue[
            tuple[ReqId, ReqMeta, LayerName, Optional[torch.cuda.Event]]
        ]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        backend = get_attn_backend(self.model_config.get_head_size(),
                                   self.model_config.dtype,
                                   self.cache_config.cache_dtype,
                                   self.block_size,
                                   use_mla=self.use_mla)
        self.backend_name = backend.get_name()
        attn_backend = backend_name_to_enum(self.backend_name)
        self._use_flashinfer = attn_backend == _Backend.FLASHINFER
        self._use_pallas = attn_backend == _Backend.PALLAS
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        self.xfer_stats = NixlKVConnectorStats()

        self._connector_metadata: \
            Optional[NixlLayerwiseConnectorMetadata] = None
        self.meta_server_client = httpx.Client(
            limits=httpx.Limits(max_connections=100000),
            timeout=None
        ) if self.tp_rank == 0 else None
        self.meta_server_executor = ThreadPoolExecutor(
            max_workers=32,
            thread_name_prefix="vllm-nixl-layerwise-meta-server")

        # Background thread for sending or receiving KV caches.
        self.kv_layerwise_send_thread: \
            Optional[KVCacheLayerwiseSendThread] = None
        self.kv_layerwise_recv_thread: \
            Optional[KVCacheLayerwiseRecvThread] = None

        # Avoid conflicts between communication ports and handshake ports.
        self._port_offset = _WORKER_RECV_PORT_OFFSET - _WORKER_BASE_PORT_OFFSET

        # set device
        _local_rank = get_world_group().local_rank
        self.device = torch.device(f"gcu:{_local_rank}")
        logger.info("Init NixlLayerwiseConnectorWorker for %s",
                    self.device)
        current_platform.set_device(self.device)


    @staticmethod
    def _nixl_handshake_listener(metadata: NixlAgentMetadata,
                                 ready_event: threading.Event, base_port: int,
                                 tp_rank: int):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded NixlAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, base_port + tp_rank)
        logger.info("Nixl-handshake starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                if msg != GET_META_MSG:
                    logger.warning(
                        "Connection listener got unexpected message %s", msg)
                sock.send_multipart((identity, b"", encoded_data))

    def _nixl_handshake(
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
        # push to. With homogeneous TP it happens to be the same rank_i.
        tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
        remote_rank = self.tp_rank // tp_ratio
        path = make_zmq_path("tcp", host, port + remote_rank)
        logger.info("Querying metadata on path: %s at remote rank %s", path,
                     remote_rank)

        # Send query for the request.
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.info("NIXL handshake: get metadata took: %s",
                         got_metadata_time - start_time)

            # Ensure engine id matches.
            if metadata.engine_id != expected_engine_id:
                raise RuntimeError(f"Remote NIXL agent engine ID mismatch. "
                                   f"Expected {expected_engine_id},"
                                   f"received {metadata.engine_id}.")

            # Register Remote agent.
            remote_agent_name = self.add_remote_agent(metadata, remote_rank,
                                                      remote_tp_size)
            setup_agent_time = time.perf_counter()
            logger.info("NIXL handshake: add agent took: %s",
                         setup_agent_time - got_metadata_time)

        # Remote rank -> agent name.
        return {remote_rank: remote_agent_name}


    def _background_nixl_handshake(self, req_id: str,
                                   remote_engine_id: EngineId, meta: ReqMeta,
                                   layer_name: LayerName,
                                   event: Optional[torch.cuda.Event]):
        # Do NIXL handshake in background and add to _ready_requests when done.
        fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake, meta.remote_host, meta.remote_port,
                meta.remote_tp_size, remote_engine_id)
            self._handshake_futures[remote_engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            fut.add_done_callback(done_callback)

        # TODO: handle failure state of future in the
        # callback, we want to fail the request in this case.
        def request_ready(_f: Future[Any], entry=(req_id, meta,
                                                  layer_name, event)):
            self._ready_requests.put(entry)

        fut.add_done_callback(request_ready)

    def _get_layer_name_prefix(self, layer_name: str):
        # for layer name such as model.layers.0.self_attn.attn
        # return model.layers.0
        parts = layer_name.split('.')
        if len(parts) >= 3:
            return '.'.join(parts[:3])
        else:
            raise ValueError(f"Invalid layer_name {layer_name}")


    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches with layerwise support"""
        xfer_buffers = kv_caches

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, ",
            self.use_mla, self.kv_buffer_device)

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
        self.total_num_layers = len(xfer_buffers.keys())
        self.total_layer_names = list(xfer_buffers.keys())
        self.layer_prefix_to_kv_tensor_names: \
            dict[str, list[str]] = defaultdict(list)

        for i, (layer_name, cache_or_caches) in enumerate(xfer_buffers.items()):
            cache_list = cache_or_caches if split_k_and_v else [
                cache_or_caches
            ]

            prefix = self._get_layer_name_prefix(layer_name)
            self.layer_prefix_to_kv_tensor_names[prefix].append(layer_name)

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
                logger.debug("register_kv_caches, layer_name: %s, "
                             "kv shape: %s.", layer_name, cache.shape)

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
        logger.debug("KV cache info, num_layers: %s, total_num_layers: %s, "
                     "num_regions: %s, total_layer_names: %s",
                     self.num_layers, self.total_num_layers,
                     self.num_regions, self.total_layer_names)
        logger.debug("KV cache info, layer_prefix_to_kv_tensor_names: %s, ",
                     self.layer_prefix_to_kv_tensor_names)


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

        # Register local/src descr for NIXL xfer.
        blocks_data = []
        for i, base_addr in enumerate(seen_base_addresses):
            kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
            # NOTE With heter-TP, more blocks are prepared than what are
            # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
            # could create fewer, but then _get_block_descs_ids needs to
            # select agent_meta.num_blocks instead of self.num_blocks for
            # local descr, and that makes handling regular flow less clean.
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len_per_layer[i]
                addr = base_addr + block_offset
                # (addr, len, device id)
                blocks_data.append((addr, kv_block_len, self.tp_rank))

            if self._use_flashinfer:
                # Separate and interleave K/V regions to maintain the same
                # descs ordering. This is needed for selecting contiguous heads
                # when split across TP ranks.
                for block_id in range(self.num_blocks):
                    block_offset = block_id * self.block_len_per_layer[i]
                    addr = base_addr + block_offset
                    # Register addresses for V cache (K registered first).
                    v_addr = addr + kv_block_len
                    blocks_data.append((v_addr, kv_block_len, self.tp_rank))
        logger.debug("Created %s blocks for src engine %s and rank %s",
                     len(blocks_data), self.engine_id, self.tp_rank)

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data,
                                                 self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs)

        # After KV Caches registered, listen for new connections.
        metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            attn_backend_name=self.backend_name,
            kv_cache_layout=self.kv_cache_layout,
            layer_names=self.total_layer_names)

        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="nixl_handshake_listener")
        self._nixl_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

        if self.vllm_config.kv_transfer_config.is_kv_producer:
            ready_event = threading.Event()
            self.kv_layerwise_send_thread = KVCacheLayerwiseSendThread(
                num_layers = self.total_num_layers,
                ready_event = ready_event,
                device = self.device,
                nixl_wrapper = self.nixl_wrapper,
                callback_func = self.send_done_sending_signal,
            )
            self.kv_layerwise_send_thread.start()
            ready_event.wait()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            ready_event = threading.Event()
            self.kv_layerwise_recv_thread = KVCacheLayerwiseRecvThread(
                recv_thread_port = self.side_channel_port + self._port_offset,
                tp_size = self.tp_size,
                tp_rank = self.tp_rank,
                ready_event = ready_event,
            )
            self.kv_layerwise_recv_thread.start()
            ready_event.wait()


    def add_remote_agent(self,
                         nixl_agent_meta: NixlAgentMetadata,
                         remote_tp_rank: int = 0,
                         remote_tp_size: int = 1) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for writing cache
        blocks to remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to write to remote rank_i.
        The latter, assuming P.world_size > D.world_size, requires that two or
        more local TP worker share the xfer from a single TP worker.

        Here's an example (non-MLA case):

        rank_offset     d_remote_tp_rank
        (kv split no)
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Prefill TP workers                     Decoder TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2

        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads//tp_ratio, block_size, head_dim]
        then D-Worker_j has [2, num_blocksD, kv_heads, block_size, head_dim]. Mind the "HND" layout format.
        Assuming num_blocksD >= num_blocksP, P-Worker0 writes to D-Worker0 by preparing the kv_heads//tp_ratio
        first heads from all the slots of all the blocks. P-Worker1 will do the same, but writing the second split
        along the kv_heads dimension, and so forth until "tp_ratio" P TP workers have wrote to D-Worker0.

        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" P TP workers.
        """ # noqa: E501
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

        # Number of P TP workers writing to a single D TP worker. This is
        # 1 when P and D `--tensor-parallel-size` match.
        tp_ratio = divide(self._tp_size[self.engine_id],
                          self._tp_size[engine_id])
        assert tp_ratio > 0, "Prefill TP cannot be smaller than decode TP"
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
                "Remote D worker KV layer cache must be of shape [2, N, "
                "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
            )

        assert self.block_size == remote_block_size, (
            "Remote D worker with different page/block size is not supported "
            f"block_size: {self.block_size}, "
            f"remote_block_size: {remote_block_size}")

        # Create dst descs and xfer side handles. TP workers have same #blocks.
        if engine_id in self.dst_num_blocks:
            assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks
        else:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        blocks_data = []
        # With homogeneous TP, P push the whole kv cache from corresponding
        # rank. With heterogeneous TP, prepare the descriptors by splitting the
        # D KV cache along kv_head dim, of P worker's kv_head size (P>D).
        # Eg. PTP2 DTP1 => D0 KV:[block0-KV_0 | block0-KV_1..].
        self.kv_caches_base_addr[
            engine_id] = nixl_agent_meta.kv_caches_base_addr

        assert len(nixl_agent_meta.kv_caches_base_addr) == len(
            self.block_len_per_layer)
        # Register all remote blocks, but only the corresponding kv heads.
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
            rank_offset = self.tp_rank % tp_ratio * kv_block_len \
                if not (self.use_mla or is_kv_replicated) else 0
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_lens[i]
                # For each block, grab the heads chunk belonging to rank_i
                # of size remote_nheads // tp_ratio, which correspond to
                # self.block_len == remote_block_len//tp_ratio bytes.
                addr = base_addr + block_offset + rank_offset
                # (addr, len, device id)
                blocks_data.append((addr, kv_block_len, remote_tp_rank))

            if self._use_flashinfer:
                # With FlashInfer index V separately to allow head splitting.
                for block_id in range(nixl_agent_meta.num_blocks):
                    block_offset = block_id * nixl_agent_meta.block_lens[i]
                    addr = base_addr + block_offset + rank_offset
                    v_addr = addr + nixl_agent_meta.block_lens[i] // 2
                    blocks_data.append((v_addr, kv_block_len, remote_tp_rank))

        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and "
            "local rank %s", len(blocks_data), engine_id, remote_tp_rank,
            self.tp_rank)

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data,
                                                 self.nixl_memory_type)
        self.dst_xfer_side_handles[
            engine_id] = self.nixl_wrapper.prep_xfer_dlist(
                remote_agent_name, descs)

        return remote_agent_name


    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        done_sending = \
            self.kv_layerwise_send_thread.get_and_clear_finished_requests() \
            if self.vllm_config.kv_transfer_config.is_kv_producer else set()
        done_recving = \
            self.kv_layerwise_recv_thread.get_and_clear_finished_requests() \
            if self.vllm_config.kv_transfer_config.is_kv_consumer else set()
        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving", self.tp_rank,
                len(done_sending), len(done_recving))
            for _ in done_sending:
                self.xfer_stats.record_transfer()

        return done_sending, done_recving

    def _meta_server(self, url, message):
        success = False
        retry = 0
        while retry < 3 and success is False:
            retry += 1
            try:
                self.meta_server_client.post(url, json=message)
                success = True
            except Exception as e:
                logger.error("Failed to connect to metaserver: %s, "
                             "retry %s time.", url, retry)
                if retry == 3:
                    raise e

    def start_load_kv(self, metadata: NixlLayerwiseConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        # Store metadata reference for save_kv_layer
        self._connector_metadata = metadata

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            assert self.kv_layerwise_recv_thread is not None
            for req_id, meta in metadata.requests.items():
                if self.tp_rank % self.tp_size == 0:
                    logger.debug("Send request %s to proxy meta-server %s",
                                 req_id, meta.meta_server)

                    kv_transfer_params = dict(
                        request_id=req_id,
                        do_remote_prefill=False,
                        do_remote_decode=True,
                        remote_block_ids=meta.local_block_ids,
                        remote_engine_id=self.engine_id,
                        remote_host=self.side_channel_host,
                        remote_port=self.side_channel_port,
                        remote_tp_size=self.tp_size,
                    )
                    future = self.meta_server_executor.submit(
                        self._meta_server,
                        url=meta.meta_server,
                        message=kv_transfer_params,
                    )

                    def handle_exception(future):
                        if future.exception():
                            logger.error(
                                f"Access meta-server fail: {future.exception()}"
                            )

                    future.add_done_callback(handle_exception)

                self.kv_layerwise_recv_thread.add_task_trace(req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for a specific layer to complete loading"""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                     attn_metadata: "AttentionMetadata",
                     connector_metadata: NixlLayerwiseConnectorMetadata,
                     **kwargs) -> None:
        """Layerwise WRITE mode: initiate send for specific layer"""
        if not self.vllm_config.kv_transfer_config.is_kv_producer:
            return

        if len(connector_metadata.requests.keys()) == 0:
            return

        assert self.kv_layerwise_send_thread is not None

        event = torch.cuda.Event()
        event.record(current_stream())
        # event  = None
        # current_stream().synchronize()

        for req_id, req_meta in connector_metadata.requests.items():
            remote_engine_id = req_meta.remote_engine_id
            logger.debug(
                "save_kv_layer for request %s layer %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                layer_name, remote_engine_id, len(req_meta.local_block_ids),
                len(req_meta.remote_block_ids))
            if remote_engine_id is None:
                logger.debug("Skip save_kv_layer for "
                             "prefill only request: %s", req_id)
                continue

            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(
                            req_id, remote_engine_id, req_meta,
                            layer_name, event)
                        continue

            # Handshake already completed, start async read xfer.
            self._write_layer_async(req_id, req_meta, layer_name, event)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._write_layer_async(*self._ready_requests.get_nowait())

    def _write_layer_async(self, req_id: ReqId, req_meta: ReqMeta,
                           layer_name: LayerName,
                           event: Optional[torch.cuda.Event]):

        # Partial prefix cache hit: just write uncomputed blocks.
        local_block_ids = req_meta.local_block_ids
        num_local_blocks = len(local_block_ids)
        num_remote_blocks = len(req_meta.remote_block_ids)
        assert num_remote_blocks <= num_local_blocks, \
            f"_write_layer_async get mismatch block size. " \
            f"P size: {num_local_blocks}, D size: {num_remote_blocks}"
        if num_remote_blocks < num_local_blocks:
            local_block_ids = local_block_ids[-num_remote_blocks:]

        # Use layer-specific handles
        local_handle = self.src_xfer_side_handle
        remote_handle = self.dst_xfer_side_handles[req_meta.remote_engine_id]

        prefix = self._get_layer_name_prefix(layer_name)
        kv_tensor_names = self.layer_prefix_to_kv_tensor_names[prefix]
        for kv_tensor_name in kv_tensor_names:
            layer_idx = self.total_layer_names.index(kv_tensor_name)

            local_block_descs_ids = self._get_block_descs_ids(
                self.engine_id, local_block_ids, layer_idx=layer_idx)
            remote_block_descs_ids = self._get_block_descs_ids(
                req_meta.remote_engine_id, req_meta.remote_block_ids,
                layer_idx=layer_idx)

            transfer_meta = LayerTransferMeta(
                request_id = req_id,
                req_meta = req_meta,
                layer_index = layer_idx,
                local_handle = local_handle,
                remote_handle = remote_handle,
                local_block_descs_ids = local_block_descs_ids,
                remote_block_descs_ids = remote_block_descs_ids,
                xfer_handler = None,
                event = event,
                expiration_time = time.perf_counter() \
                    + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
            )
            self.kv_layerwise_send_thread.add_transfer_meta(transfer_meta)
            logger.debug("Started layerwise WRITE: req %s layer %s, "
                         "transfer_meta: %s", req_id,
                         kv_tensor_name, transfer_meta)


    def send_done_sending_signal(self, transfer_meta: LayerTransferMeta):
        req_id = transfer_meta.request_id
        req_meta = transfer_meta.req_meta

        try:
            tp_ratio = self._tp_size[self.engine_id] // req_meta.remote_tp_size
            remote_rank = self.tp_rank // tp_ratio
            remote_port = req_meta.remote_port + self._port_offset + remote_rank
            logger.debug("Sending done sending signal for request %s to %s:%d",
                         req_id, req_meta.remote_host, remote_port)
            path = make_zmq_path("tcp", req_meta.remote_host, remote_port)
            notif_msg = f"{req_id}:{tp_ratio}"
            with zmq_ctx(zmq.REQ, path) as sock:
                msg_encoder = msgspec.msgpack.Encoder()
                encoded_data = msg_encoder.encode((DONE_SENDING_MSG, notif_msg))
                sock.send(encoded_data)
                ack = sock.recv()
                if ack != b"ACK":
                    raise ValueError(f"Unexpected ACK response: {ack}")
                logger.debug("NIXL layerwise, send_done_sending_signal "
                             "for request: %s", req_id)

        except Exception as e:
            logger.error("Sending done sending signal for request %s to %s:%s "
                         "fail with error: %s", req_id,
                         req_meta.remote_host, remote_port)

    def _get_block_descs_ids(self,
                             engine_id: str,
                             block_ids: list[int],
                             layer_idx: Optional[int] = None) -> np.ndarray:
        """
        Get the descs ids for a set of block ids.
        If layer_idx is provided, we use the region_ids for the given layer.
        Otherwise, we use all regions.
        """
        if layer_idx is None:
            region_ids = np.arange(self.num_regions)
        else:
            assert layer_idx < self.total_num_layers
            if self.total_num_layers < self.num_regions:
                # If we have more regions than layers, we assume that
                # the regions are organized as [K0, V0, K1, V1, ...]
                # and we select K_i and V_i
                assert 2 * self.total_num_layers == self.num_regions
                region_ids = np.arange(2 * layer_idx, 2 * layer_idx + 2)
            else:
                # Otherwise, we assume we have MLA and select i-th layer
                assert self.total_num_layers == self.num_regions
                region_ids = np.arange(layer_idx, layer_idx + 1)

        num_blocks = self.dst_num_blocks[engine_id]

        # Compute the desc ids for each block.
        region_ids = region_ids[:, None]
        block_ids = np.array(block_ids)[None, :]
        descs_ids = region_ids * num_blocks + block_ids
        return descs_ids.flatten()

    def get_backend_aware_kv_block_len(self, layer_idx: int):
        """
        Get the block length for one K/V element (K and V have the same size).

        For FA and other backends, this is equal to the length of the whole
        block, as K and V are in separate regions.
        For FlashInfer, this is half the length of the whole block, as K and V
        share the same region.
        """
        if self._use_flashinfer:
            # For indexing only half (either just the K or V part).
            block_len = self.block_len_per_layer[layer_idx] // 2
        else:
            block_len = self.block_len_per_layer[layer_idx]
        return block_len

    def get_kv_connector_stats(self) -> Optional[KVConnectorStats]:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def shutdown(self):
        """Shutdown the connector worker."""
        self._handshake_initiation_executor.shutdown(wait=False)
        self.meta_server_executor.shutdown(wait=False)
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join(timeout=0)
            self._nixl_handshake_listener_t = None
        if self.kv_layerwise_send_thread is not None:
            self.kv_layerwise_send_thread.join(timeout=0)
            self.kv_layerwise_send_thread = None
        if self.kv_layerwise_recv_thread is not None:
            self.kv_layerwise_recv_thread.join(timeout=0)
            self.kv_layerwise_recv_thread = None

        # Clean up descriptor lists
        if self.src_xfer_side_handle:
            self.nixl_wrapper.release_dlist_handle(self.src_xfer_side_handle)
            self.src_xfer_side_handle = 0
        for dst_xfer_side_handle in self.dst_xfer_side_handles.values():
            self.nixl_wrapper.release_dlist_handle(dst_xfer_side_handle)
        self.dst_xfer_side_handles.clear()
        for remote_agents in self._remote_agents.values():
            for agent_name in remote_agents.values():
                self.nixl_wrapper.remove_remote_agent(agent_name)
        self._remote_agents.clear()
        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()

        logger.info("NixlLayerwiseConnectorWorker shutdown complete")

@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(ctx=ctx,
                              path=addr,
                              socket_type=socket_type,
                              bind=socket_type == zmq.ROUTER)
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if "num_successful_transfers" not in self.data:
            self.data["num_successful_transfers"] = 0

    def reset(self):
        self.data = {"num_successful_transfers": 0}

    def record_transfer(self):
        # TODO: record actual transfer stats when available
        self.data["num_successful_transfers"] += 1

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.data["num_successful_transfers"] == 0

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            self.data["num_successful_transfers"] += other.data[
                "num_successful_transfers"]
        return self

    def reduce(self) -> dict[str, Union[int, float]]:
        # TODO: reduce stats to a single value, calculate latency/throughput
        return {
            "num_successful_transfers": self.data["num_successful_transfers"]
        }
