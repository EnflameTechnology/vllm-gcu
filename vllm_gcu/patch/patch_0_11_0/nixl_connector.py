# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Optional

from vllm.logger import init_logger
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnectorScheduler, NixlConnectorWorker
from vllm_gcu.utils import get_tx_ctx, get_tx_mark_func
import vllm_gcu.envs as gcu_envs
import vllm.envs as envs
from unittest.mock import patch
import orjson
import time

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

origin_get_num_new_matched_tokens = NixlConnectorScheduler.get_num_new_matched_tokens


def get_num_new_matched_tokens(self, request: "Request",
                               num_computed_tokens: int) -> tuple[int, bool]:
    ret = origin_get_num_new_matched_tokens(self, request, num_computed_tokens)
    params = request.kv_transfer_params
    if params is not None and params.get("do_remote_prefill") and \
            gcu_envs.VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE:
        logger.debug("VLLM_GCU_NIXL_ENABLE_FIRST_TOKEN_REUSE is enabled,"
                     "skipping first token")
        first_token = params.get("first_token")
        if first_token:
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
        if request.num_output_tokens > 0:
            first_token = request.output_token_ids[0]
        else:
            logger.debug("No output tokens for request %s", request.request_id)
        txfer_params['first_token'] = first_token
    return async_save, txfer_params


origin_read_blocks = NixlConnectorWorker._read_blocks

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