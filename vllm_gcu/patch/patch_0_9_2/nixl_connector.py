# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import zmq

from vllm import envs
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.distributed.utils import divide
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend
from vllm.utils import make_zmq_path, make_zmq_socket, round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
import vllm_gcu.envs as gcu_envs
from unittest.mock import patch

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

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
        "NIXLConnector get_num_new_matched_tokens: "
        "num_computed_tokens=%s, kv_transfer_params=%s",
        num_computed_tokens, params)

    if params is not None and params.get("do_remote_prefill"):
        # Remote prefill: get all prompt blocks from remote.
        assert num_computed_tokens % self.block_size == 0
        if gcu_envs.VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER:
            count = max(request.num_prompt_tokens - num_computed_tokens, 0)
            logger.debug("VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER is enabled, count=%s", count)
        else:
            rounded_num_prompt_tokens = round_down(
                len(request.prompt_token_ids), self.block_size)
            count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)

        if gcu_envs.VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE:
            logger.debug("VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE is enabled, skipping first token")
            first_token = params.get("first_token")
            if first_token:
                logger.debug("NIXLConnector: first_token(%s) from kv_transfer_params", first_token)
                request.prompt_token_ids.append(first_token)
                request.num_prompt_tokens = len(request.prompt_token_ids)
                request._all_token_ids.append(first_token)
            else:
                logger.debug("NIXLConnector: no first_token in kv_transfer_params")

        if count > 0:
            return count, True

    # No remote prefill for this request.
    return 0, False


def request_finished(
    self,
    request: "Request",
    block_ids: list[int],
) -> tuple[bool, Optional[dict[str, Any]]]:
    """
    Once a request is finished, determine whether request blocks
    should be freed now or will be sent asynchronously and freed later.
    """

    params = request.kv_transfer_params
    logger.debug(
        "NIXLConnector request_finished, request_status=%s, "
        "kv_transfer_params=%s", request.status, params)
    if not params:
        return False, None

    if params.get("do_remote_prefill"):
        # If do_remote_prefill is still True when the request is finished,
        # update_state_after_alloc must not have been called (the request
        # must have been aborted before it was scheduled).
        # To avoid stranding the prefill blocks in the prefill instance,
        # we must add empty block_ids to _reqs_need_recv so that our
        # worker side will notify and free blocks in the prefill instance.
        self._reqs_need_recv[request.request_id] = (request, [])
        params["do_remote_prefill"] = False
        return False, None

    if (not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
        return False, None

    # Get computed blocks.
    all_full = gcu_envs.VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER or \
        request.num_computed_tokens % self.block_size == 0
    computed_block_ids = block_ids if all_full else block_ids[:-1]

    # If prompt < block_size, no xfer so free blocks immediately.
    delay_free_blocks = len(computed_block_ids) > 0

    first_token = None
    if gcu_envs.VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE:
        logger.debug("VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE is enabled")
        # Get the first token from the request's output tokens
        if request.num_output_tokens > 0:
            first_token = request.output_token_ids[0]
        else:
            logger.debug("No output tokens for request %s", request.request_id)
    logger.debug("NIXLConnector request_finished, first_token=%s", first_token)

    return delay_free_blocks, dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_block_ids=computed_block_ids,
        remote_engine_id=self.engine_id,
        remote_host=self.side_channel_host,
        remote_port=self.side_channel_port,
        tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        first_token=first_token)

patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorScheduler.get_num_new_matched_tokens", get_num_new_matched_tokens).start()
patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorScheduler.request_finished", request_finished).start()
