# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Optional

from vllm.logger import init_logger
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnectorScheduler

import vllm_gcu.envs as gcu_envs
from unittest.mock import patch

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


patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorScheduler.get_num_new_matched_tokens",
    get_num_new_matched_tokens).start()
patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnectorScheduler.request_finished",
    request_finished).start()
