# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import resource
import secrets
import signal
import traceback
import uuid
from threading import Condition
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import uvloop
from aiohttp import web
from starlette.datastructures import Headers
from transformers import AutoTokenizer

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
)

routes = web.RouteTableDef()

# Environment variable to control whether to return first chunk from prefill
# If set to "1", "true", or "yes", the first chunk from prefill will be returned
# Default: False (all chunks come from decode)
VLLM_PD_FIRST_TOKEN_FROM_P = os.environ.get("VLLM_PD_FIRST_TOKEN_FROM_P", "0").lower() in ("1", "true", "yes")


# Environment variable to control whether to enable API key authentication
# If set to "true", "yes", or "1", API key authentication will be enabled
# Default: False (no authentication required)
VLLM_ROUTER_ENABLE_API_KEY = os.environ.get("VLLM_ROUTER_ENABLE_API_KEY", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Environment variable to control whether to enable DPLB (Disaggregated Prefill Load Balancer)
# If set to "true", "yes", or "1", DPLB will be enabled for prefill client selection
# Default: False (use round-robin load balancing)
VLLM_ROUTER_ENABLE_DPLB = os.environ.get("VLLM_ROUTER_ENABLE_DPLB", "false").lower() in ("1", "true", "yes")

# Environment variable to specify the DPLB tokenizer path. If it is None,
# the length of the request is used to approximate the token length.
VLLM_ROUTER_DPLB_TOKENIZER_PATH = os.environ.get("VLLM_ROUTER_DPLB_TOKENIZER_PATH", None)

# Environment variable to control whether to enable layerwise transfer
# If set to "true", "yes", or "1", layerwise will be enabled for nixl connector
# Default: False
VLLM_TRANSFER_ENABLE_LAYERWISE = os.environ.get("VLLM_TRANSFER_ENABLE_LAYERWISE", "false").lower() in ("1", "true", "yes")


class TokenManager:
    def __init__(self, api_keys: Optional[List[str]] = None, enable_auth: bool = False):
        """
        Initialize token manager with API keys.
        Args:
            api_keys: List of API keys. If None or empty, verification is disabled.
            enable_auth: Whether to enable authentication. Controlled by VLLM_ROUTER_ENABLE_API_KEY.
        """
        self.api_tokens = []
        if api_keys:
            for key in api_keys:
                self.api_tokens.append(hashlib.sha256(key.encode("utf-8")).digest())
        # Only enable if both: 1) env var is true AND 2) api_keys are provided
        self.enabled = enable_auth and len(self.api_tokens) > 0

    def verify_token(self, headers: Headers) -> bool:
        """
        Verify the bearer token from Authorization header.
        Returns:
            True if token is valid or verification is disabled, False otherwise.
        """
        # If no tokens configured, allow all requests
        if not self.enabled:
            return True

        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()
        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match


def is_chat_api(path: str) -> bool:
    if not path:
        return False
    normalized = urlparse(path).path.rstrip("/")
    return normalized == "/v1/chat/completions"


def verify_api_key(request: web.Request):
    """
    Dependency to verify API key from request headers.
    Raises HTTPException if verification fails.
    """
    global token_manager
    if not token_manager.verify_token(request.headers):
        raise web.HTTPException(
            status=401,
            text="Invalid or missing API key\n",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Module-level constants for better maintainability
ALLOWED_POST_HEADERS: Tuple[str, ...] = ("User-Agent", "Accept", "Content-Type")
ALLOWED_RESPONSE_HEADERS: Tuple[str, ...] = ("Content-Type", "Transfer-Encoding")


def generate_headers(headers: Dict[str, Any], header_type: Literal["POST", "RESPONSE"]) -> Dict[str, str]:
    """
    Generate filtered headers based on request/response type.

    Args:
        headers: Source headers dictionary
        header_type: "POST" for request headers, "RESPONSE" for response headers

    Returns:
        Filtered dictionary containing only allowed headers with non-None values

    Raises:
        ValueError: If header_type is not "POST" or "RESPONSE"
    """
    if header_type == "POST":
        allowed = ALLOWED_POST_HEADERS
    elif header_type == "RESPONSE":
        allowed = ALLOWED_RESPONSE_HEADERS
    else:
        raise ValueError(f"Invalid header_type: {header_type}")

    filtered_headers = {k: v for k in allowed if (v := headers.get(k)) is not None}
    filtered_headers["Authorization"] = f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    return filtered_headers


def check_nested_dict(data: dict, *keys, path="response_json"):
    """
    Check nested dictionaries, returning the False if any key does not exist

    Args:
        data: The dictionary to access
        *keys: Sequence of keys to traverse
        path: Path string used in log messages for context
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and key < len(current):
            current = current[key]
        else:
            logger.info("Missing key in %s: %s", path, (keys[: keys.index(key) + 1]))
            return False
    return True


class LLMInstance:
    def __init__(self, id, ip, port):
        self.id = id
        self.endpoint = f"http://{ip}:{port}"

        self.token_pressure = 0
        self.request_count = 0

    def __lt__(self, other):
        return self.token_pressure < other.token_pressure

    def __repr__(self):
        return f"LLMInstance<id={self.id}, endpoint={self.endpoint}>"

    def post_request(self, request: web.Request, headers, payload):
        return client_session.post(
            f"{self.endpoint}{request.rel_url}",
            headers=headers,
            json=payload,
        )


class TokenLoadBalancer:
    def __init__(self):
        self.llm_instances = []
        self.condition = Condition()
        self.enable = VLLM_ROUTER_ENABLE_DPLB
        self.tokenizer = None
        if self.enable and VLLM_ROUTER_DPLB_TOKENIZER_PATH is not None:
            if os.path.exists(VLLM_ROUTER_DPLB_TOKENIZER_PATH):
                self.tokenizer = AutoTokenizer.from_pretrained(VLLM_ROUTER_DPLB_TOKENIZER_PATH, use_fast=True)
            else:
                raise ValueError(f"Tokenizer path {VLLM_ROUTER_DPLB_TOKENIZER_PATH} is not exist.")

        self.round_robin_index = 0
        self.num_llm_instances = 0

    def add(self, llm_instance):
        with self.condition:
            self.llm_instances.append(llm_instance)
            self.num_llm_instances = len(self.llm_instances)
            logger.info(f"Add LLMInstance: {llm_instance}")

    def remove(self, llm_instance):
        try:
             with self.condition:
                 self.llm_instances.remove(llm_instance)
                 self.num_llm_instances = len(self.llm_instances)
                 logger.info(f"Remove LLMInstance: {llm_instance}")
        except ValueError:
            pass

    @contextlib.contextmanager
    def get_next(self, req_data: Dict[str, Any] = None, message_size: int = 1) -> Generator[LLMInstance, None, None]:
        if not self.llm_instances:
            raise ValueError("Registered LLM instances is empty.")
        if not self.enable:
            with self.condition:
                instance = self.llm_instances[self.round_robin_index % self.num_llm_instances]
                self.round_robin_index += 1
                if self.round_robin_index == self.num_llm_instances:
                    self.round_robin_index = 0
            try:
                yield instance
            finally:
                pass
        else:
             with self.condition:
                 instance = min(self.llm_instances)
             token_pressure = self.get_token_num(req_data, message_size)
             try:
                 instance.token_pressure += token_pressure
                 instance.request_count += 1
                 yield instance
             finally:
                 instance.token_pressure -= token_pressure
                 instance.request_count -= 1

    def get_token_num(self, req_data: Dict[str, Any], message_size: int = 1) -> int:
        if self.tokenizer is None or req_data is None:
            return message_size if self.enable else 1
        if "messages" in req_data:
            total = 3  # offset for openai
            for m in req_data["messages"]:
                total += len(self.tokenizer.encode(m["role"], add_special_tokens=False))
                total += len(self.tokenizer.encode(m["content"], add_special_tokens=False))
            return total
        if "prompt" in req_data:
            if isinstance(req_data["prompt"], list):
                prompt = "\n".join(req_data["prompt"])
            else:
                prompt = req_data["prompt"]
            return len(self.tokenizer.encode(prompt, add_special_tokens=True))
        raise ValueError(f"TokenLoadBalancer get not supported req_data: {req_data}.")


async def server_response(
    request: web.Request, response: aiohttp.ClientResponse, request_id: str = "none", is_chat=True, response_json=None
):
    is_stream = response.headers.get("Transfer-Encoding") == "chunked"
    resp = web.StreamResponse() if is_stream else web.Response()

    resp.set_status(response.status, response.reason)
    resp.headers.update(generate_headers(response.headers, "RESPONSE"))
    # logger.info("Server response, is_stream: %s, Content-Type: %s.", is_stream, response.headers.get("Content-Type"))
    is_decode_first_chunk = True
    try:
        if not is_stream:
            resp.body = await response.read()
        else:
            await resp.prepare(request)

            if VLLM_TRANSFER_ENABLE_LAYERWISE and VLLM_PD_FIRST_TOKEN_FROM_P:
                event = await proxy_state.get_event(request_id)
                if event:
                    await event.wait()
                    response_json = await proxy_state.get_response_json(request_id)
            if (response_json is not None) and VLLM_PD_FIRST_TOKEN_FROM_P:
                if is_chat:
                    # For chat completions, use delta format
                    has_content = check_nested_dict(
                        response_json, "choices", 0, "message", "content", path="response_json"
                    )
                    has_reasoning_content = check_nested_dict(
                        response_json, "choices", 0, "message", "reasoning_content", path="response_json"
                    )
                    if not (has_content or has_reasoning_content):
                        raise ValueError("Either content or reasoning_content must be present.")
                    first_chunk = {
                        "id": response_json["id"],
                        "object": "chat.completion.chunk",
                        "created": response_json["created"],
                        "model": response_json["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": (
                                        response_json["choices"][0]["message"]["content"] if has_content else None
                                    ),
                                    "reasoning_content": (
                                        response_json["choices"][0]["message"]["reasoning_content"]
                                        if has_reasoning_content
                                        else None
                                    ),
                                },
                                "logprobs": None,
                                "finish_reason": None,
                                "token_ids": None,
                            }
                        ],
                    }
                else:
                    # For regular completions, use text format
                    has_text = check_nested_dict(response_json, "choices", 0, "text", path="response_json")
                    if not has_text:
                        raise ValueError("text must be present.")
                    first_chunk = {
                        "id": response_json["id"],
                        "object": "text_completion",
                        "created": response_json["created"],
                        "model": response_json["model"],
                        "choices": [
                            {
                                "index": 0,
                                "text": response_json["choices"][0]["text"] if has_text else None,
                                "logprobs": response_json["choices"][0].get("logprobs"),
                                "finish_reason": None,
                                "stop_reason": None,
                                "prompt_token_ids": None,
                                "token_ids": None,
                            }
                        ],
                        "usage": None,
                    }
                first_chunk_data = (
                    "data: " + json.dumps(first_chunk, separators=(",", ":"), ensure_ascii=False) + "\n\n"
                )
                first_chunk_resp = first_chunk_data.encode()
                logger.debug("First token from prefill, request_id: %s, first_chunk: %s", request_id, first_chunk_data)
                await resp.write(first_chunk_resp)

            chunks = []
            async for chunk, end_of_chunk in response.content.iter_chunks():
                logger.debug("Recv forward response, request_id: %s", request_id)
                # The first token returned by the '/v1/completions' endpoint is crucial and cannot be skipped.
                # Perhaps '/v1/completions' and '/v1/chat/completions' differ in some ways.
                if is_decode_first_chunk and is_chat and VLLM_PD_FIRST_TOKEN_FROM_P:
                    is_decode_first_chunk = False
                    logger.debug("Request %s, skip first chunk: %s", request_id, chunk)
                    continue
                is_decode_first_chunk = False
                chunks.append(chunk)
                if end_of_chunk:
                    await resp.write(b"".join(chunks))
                    chunks.clear()

    except Exception as e:
        logger.info(f"Exception [{e}] occurred while server response {request_id}")
        pass
    return resp


async def _handle_completions(request: web.Request):
    headers = generate_headers(request.headers, "POST")
    verify_api_key(request)
    try:
        req_data_bin = await request.read()
        req_data = json.loads(req_data_bin)

        max_tokens = int(req_data.pop("max_tokens", 0) or 0)
        if "max_completion_tokens" in req_data:
            max_tokens = int(req_data.pop("max_completion_tokens") or 0)
        if max_tokens:
            req_data["max_tokens"] = max_tokens
    except:
        return web.Response(status=400, text="Invalid request\n")

    request_id = req_data.setdefault("request_id", request.headers.get("X-Request-Id"))
    if request_id is None:
        request_id = req_data.get("query_id", str(uuid.uuid4()))
        req_data["request_id"] = request_id

    is_stream = req_data.get("stream", False)
    is_chat = is_chat_api(request.url.path) and request.method == "POST"

    # For TokenLoadBalancer
    message_size = len(req_data_bin)

    if max_tokens == 1:
        with prefill_balancer.get_next(req_data, message_size) as prefill:
            logger.debug("[Only prefill] Send to prefill %s, req id: %s", prefill.id, request_id)
            async with prefill.post_request(request, headers, req_data) as response:
                res = await server_response(request, response, request_id, is_chat)
                logger.debug("[Only prefill] Recv from prefill %s, req id: %s", prefill.id, request_id)
                return res

    try:
        with decode_balancer.get_next() as decode:
            decode_task = None
            with prefill_balancer.get_next(req_data, message_size) as prefill:
                logger.debug(
                    f"Request [{request_id}] from {request.remote}" f" (prefill={prefill.id} --> decode={decode.id})"
                )
                prefill_req_data = {
                    **req_data,
                    "kv_transfer_params": {
                        "do_remote_decode": True,
                        "do_remote_prefill": False,
                        "remote_engine_id": None,
                        "remote_block_ids": None,
                        "remote_host": None,
                        "remote_port": None,
                    },
                    "stream": False,
                    "max_tokens": 1,
                }
                if "max_completion_tokens" in prefill_req_data:
                    prefill_req_data["max_completion_tokens"] = 1
                if "stream_options" in prefill_req_data:
                    del prefill_req_data["stream_options"]

                # 1. Send to prefill
                logger.debug("Send to prefill %s, req id: %s", prefill.id, request_id)
                async with prefill.post_request(request, headers, prefill_req_data) as prefill_response:
                    if not prefill_response.ok:
                        logger.debug("Response Not OK, prefill %s, req id: %s", prefill.id, request_id)
                    else:
                        logger.debug("Recv response status from prefill %s, req id: %s", prefill.id, request_id)

                    # 2. Response of prefill
                    await prefill_response.read()
                    response_json = await prefill_response.json()
                    # logger.debug("Get prefill response_json: %s", response_json)

                    kv_transfer_params = response_json.get("kv_transfer_params", {})
                    if kv_transfer_params:
                        req_data["kv_transfer_params"] = kv_transfer_params
                    if VLLM_PD_FIRST_TOKEN_FROM_P:
                        req_data["max_tokens"] -= 1
                    logger.debug("Recv response from prefill %s, req id: %s", prefill.id, request_id)

                    # 3. Create async task and send to decode
                    async def decode_runner():
                        decode_req_data = {
                            **req_data,
                        }
                        logger.debug("Send to decode %s, req id: %s", decode.id, request_id)
                        async with decode.post_request(request, headers, decode_req_data) as decode_response:
                            res = await server_response(request, decode_response, request_id, is_chat, response_json)
                            logger.debug("Recv from decode %s, req id: %s", decode.id, request_id)
                            return res

                    decode_task = asyncio.create_task(decode_runner())

            # 4. Response of decode
            decode_response = await decode_task
            if not is_stream and VLLM_PD_FIRST_TOKEN_FROM_P:
                decode_response_json = json.loads(decode_response.body.decode())
                if is_chat:
                    has_content = check_nested_dict(
                        response_json, "choices", 0, "message", "content", path="response_json"
                    )
                    has_reasoning_content = check_nested_dict(
                        response_json, "choices", 0, "message", "reasoning_content", path="response_json"
                    )
                    if not (has_content or has_reasoning_content):
                        raise ValueError("Either content or reasoning_content must be present.")
                    if has_content:
                        first_token = response_json["choices"][0]["message"]["content"]
                        if first_token:
                            decode_response_json["choices"][0]["message"]["content"] = (
                                first_token + decode_response_json["choices"][0]["message"]["content"]
                            )

                    if has_reasoning_content:
                        first_reasoning_token = response_json["choices"][0]["message"]["reasoning_content"]
                        if first_reasoning_token:
                            decode_response_json["choices"][0]["message"]["reasoning_content"] = (
                                first_reasoning_token
                                + decode_response_json["choices"][0]["message"]["reasoning_content"]
                            )
                else:
                    has_text = check_nested_dict(response_json, "choices", 0, "text", path="response_json")
                    if not has_text:
                        raise ValueError("text must be present.")
                    first_token = response_json["choices"][0]["text"]
                    decode_response_json["choices"][0]["text"] = (
                        first_token + decode_response_json["choices"][0]["text"]
                    )
                new_body = json.dumps(decode_response_json, ensure_ascii=False, separators=(",", ":"))
                new_headers = dict(decode_response.headers)
                return web.Response(
                    body=new_body, status=decode_response.status, reason=decode_response.reason, headers=new_headers
                )
            else:
                return decode_response

    except ValueError as e:
        logger.info(f"ValueError [{e}] occurred while processing {request_id}")
        pass
    except Exception as e:
        logger.error(f"Exception [{e}] occurred while processing {request_id}")
        traceback.print_exc()
    if decode_task and not decode_task.done():
        decode_task.cancel()
    return web.Response(status=500, text="Exception occurred in _handle_completions\n")


async def _handle_completions_layerwise(request: web.Request):
    headers = generate_headers(request.headers, "POST")
    verify_api_key(request)
    try:
        req_data_bin = await request.read()
        req_data = json.loads(req_data_bin)

        max_tokens = int(req_data.pop("max_tokens", 0) or 0)
        if "max_completion_tokens" in req_data:
            max_tokens = int(req_data.pop("max_completion_tokens") or 0)
        if max_tokens:
            req_data["max_tokens"] = max_tokens
    except:
        return web.Response(status=400, text="Invalid request\n")

    request_id = req_data.setdefault("request_id", request.headers.get("X-Request-Id"))
    if request_id is None:
        request_id = req_data.get("query_id", str(uuid.uuid4()))
        req_data["request_id"] = request_id

    is_stream = req_data.get("stream", False)
    is_chat = is_chat_api(request.url.path) and request.method == "POST"

    # For TokenLoadBalancer
    message_size = len(req_data_bin)

    if max_tokens == 1:
        with prefill_balancer.get_next(req_data, message_size) as prefill:
            logger.debug("[Only prefill] Send to prefill %s, req id: %s", prefill.id, request_id)
            async with prefill.post_request(request, headers, req_data) as response:
                res = await server_response(request, response, request_id, is_chat)
                logger.debug("[Only prefill] Recv from prefill %s, req id: %s", prefill.id, request_id)
                return res

    await proxy_state.add_reqs_stat(request_id, request, headers, req_data, is_chat, message_size)
    await proxy_state.add_event(request_id)

    try:
        with decode_balancer.get_next() as decode:
            async def decode_runner():
                decode_req_data = {
                    **req_data,
                    "kv_transfer_params": {
                        "do_remote_decode": False,
                        "do_remote_prefill": True,
                        "remote_engine_id": None,
                        "remote_block_ids": None,
                        "remote_host": None,
                        "remote_port": None,
                        "meta_server": proxy_state.meta_server,
                    },
                }
                logger.debug("Send to decode %s, req id: %s", decode.id, request_id)
                async with decode.post_request(request, headers, decode_req_data) as decode_response:
                    res = await server_response(request, decode_response, request_id, is_chat)
                    logger.debug("Recv from decode %s, req id: %s", decode.id, request_id)
                    return res
            decode_task = asyncio.create_task(decode_runner())

            # Response of decode
            decode_response = await decode_task
            if not is_stream and VLLM_PD_FIRST_TOKEN_FROM_P:
                event = await proxy_state.get_event(request_id)
                assert event is not None, f"Req {request_id} get invalid event."
                await event.wait()
                response_json = await proxy_state.get_response_json(request_id)
                assert response_json is not None, f"Req {request_id} get invalid prefill response_json."
                decode_response_json = json.loads(decode_response.body.decode())
                if is_chat:
                    has_content = check_nested_dict(
                        response_json, "choices", 0, "message", "content", path="response_json"
                    )
                    has_reasoning_content = check_nested_dict(
                        response_json, "choices", 0, "message", "reasoning_content", path="response_json"
                    )
                    if not (has_content or has_reasoning_content):
                        raise ValueError("Either content or reasoning_content must be present.")
                    if has_content:
                        first_token = response_json["choices"][0]["message"]["content"]
                        if first_token:
                            decode_response_json["choices"][0]["message"]["content"] = (
                                first_token + decode_response_json["choices"][0]["message"]["content"]
                            )

                    if has_reasoning_content:
                        first_reasoning_token = response_json["choices"][0]["message"]["reasoning_content"]
                        if first_reasoning_token:
                            decode_response_json["choices"][0]["message"]["reasoning_content"] = (
                                first_reasoning_token
                                + decode_response_json["choices"][0]["message"]["reasoning_content"]
                            )
                else:
                    has_text = check_nested_dict(response_json, "choices", 0, "text", path="response_json")
                    if not has_text:
                        raise ValueError("text must be present.")
                    first_token = response_json["choices"][0]["text"]
                    decode_response_json["choices"][0]["text"] = (
                        first_token + decode_response_json["choices"][0]["text"]
                    )
                new_body = json.dumps(decode_response_json, ensure_ascii=False, separators=(",", ":"))
                new_headers = dict(decode_response.headers)
                return web.Response(
                    body=new_body, status=decode_response.status, reason=decode_response.reason, headers=new_headers
                )

            await proxy_state.pop_response_json(request_id)
            await proxy_state.pop_event(request_id)

            return decode_response

    except ValueError as e:
        logger.info(f"ValueError [{e}] occurred while processing {request_id}")
        pass
    except Exception as e:
        logger.error(f"Exception [{e}] occurred while processing {request_id}")
        traceback.print_exc()
    if decode_task and not decode_task.done():
        decode_task.cancel()
    return web.Response(status=500, text="Exception occurred in _handle_completions_layerwise\n")


@routes.post("/v1/completions")
@routes.post("/v1/chat/completions")
async def handle_completions(request: web.Request):
    if VLLM_TRANSFER_ENABLE_LAYERWISE:
        return await _handle_completions_layerwise(request)
    else:
        return await _handle_completions(request)


def get_origin_request_id(req_id):
    if req_id.startswith("chatcmpl-"):
        return req_id.replace("chatcmpl-", "")
    elif req_id.startswith("cmpl-"):
        return req_id.replace("cmpl-", "")[:-2]
    else:
        return req_id


@routes.post("/v1/metaserver")
async def meta_server(request: web.Request):
    try:
        req_data_bin = await request.read()
        kv_transfer_params = json.loads(req_data_bin)

        request_id = kv_transfer_params["request_id"]
        request_id = get_origin_request_id(request_id)
        request, headers, req_data, is_chat, message_size = await proxy_state.get_and_pop_req(request_id)

        req_data["kv_transfer_params"] = kv_transfer_params
        prefill_req_data = {
            **req_data,
            "stream": False,
            "max_tokens": 1,
        }
        if "max_completion_tokens" in prefill_req_data:
            prefill_req_data["max_completion_tokens"] = 1
        if "stream_options" in prefill_req_data:
            del prefill_req_data["stream_options"]

        with prefill_balancer.get_next(prefill_req_data, message_size) as prefill:
            logger.debug("Send to prefill %s, req id: %s", prefill.id, request_id)
            async with prefill.post_request(request, headers, prefill_req_data) as response:
                response.raise_for_status()
                logger.debug("Recv from prefill %s, req id: %s", prefill.id, request_id)

                if not response.ok:
                    logger.debug("Response Not OK, prefill %s, req id: %s", prefill.id, request_id)
                else:
                    logger.debug("Recv response status from prefill %s, req id: %s", prefill.id, request_id)

                response_json = await response.json()
                logger.debug("Get prefill response_json: %s", response_json)
                await proxy_state.add_response_json(request_id, response_json)
                event = await proxy_state.get_event(request_id)
                if event:
                    event.set()

        return web.Response(status=200, text=f"Done neta server for {request_id}\n")

    except Exception as e:
        logger.error(f"Exception [{e}] occurred while processing meta_server")
        traceback.print_exc()
        return web.Response(status=500, text="Exception occurred in metaserver\n")


@routes.get("/healthcheck")
async def healthcheck(request: web.Request):
    """Simple endpoint to check if the server is running."""
    prefill_num = len(prefill_balancer.llm_instances)
    decode_num = len(decode_balancer.llm_instances)
    if prefill_num == 0 or decode_num == 0:
        return web.Response(
            status=500,
            text=f"LLM instance error. There are {prefill_num} prefill_instances "
            f"and {decode_num} decode_instances\n",
        )
    return web.Response(
        status=200, text=f"Health check OK. There are {prefill_num} prefills and {decode_num} decodes\n"
    )

class ProxyState:
    def __init__(self):
        self.meta_server = None

        self._reqs_stat_lock = asyncio.Lock()
        self.reqs_stat_dict = {}

        self._prefill_event_lock = asyncio.Lock()
        self.prefill_events = {}

        self._prefill_response_lock = asyncio.Lock()
        self.prefill_response_json = {}


    async def register_meta_server(self, host, port):
        self.meta_server = f"http://{host}:{port}/v1/metaserver"
        logger.info("Register meta server: %s", self.meta_server)

    async def add_reqs_stat(self, req_id, request, headers, req_data, is_chat, message_size):
        async with self._reqs_stat_lock:
            self.reqs_stat_dict[req_id] = (request, headers, req_data, is_chat, message_size)

    async def get_and_pop_req(self, req_id):
        async with self._reqs_stat_lock:
            assert req_id in self.reqs_stat_dict, f"Request {req_id} not in process."
            values = self.reqs_stat_dict[req_id]
            self.reqs_stat_dict.pop(req_id)
            return values

    async def add_event(self, req_id):
        async with self._prefill_event_lock:
            self.prefill_events[req_id] = asyncio.Event()

    async def get_event(self, req_id):
        async with self._prefill_event_lock:
            if req_id in self.prefill_events:
                return self.prefill_events[req_id]
            else:
                return None

    async def pop_event(self, req_id):
        async with self._prefill_event_lock:
            if req_id in self.prefill_events:
                self.prefill_events.pop(req_id)

    async def add_response_json(self, req_id, response_json):
        async with self._prefill_response_lock:
            self.prefill_response_json[req_id] = response_json

    async def get_response_json(self, req_id):
        async with self._prefill_response_lock:
            if req_id in self.prefill_response_json:
                return self.prefill_response_json[req_id]
            else:
                return None

    async def pop_response_json(self, req_id):
        async with self._prefill_response_lock:
            if req_id in self.prefill_response_json:
                self.prefill_response_json.pop(req_id)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--api-keys",
        "--api-key",
        type=str,
        nargs="+",
        default=None,
        help="API keys for authentication. If not provided, no authentication is required.",
    )

    # For prefiller instances
    parser.add_argument("--prefiller-hosts", "--prefiller-host", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100])

    # For decoder instances
    parser.add_argument("--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200])

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    if VLLM_ROUTER_ENABLE_API_KEY and (args.api_keys is None) and (args.api_key is None):
        raise ValueError("Must set api_key when VLLM_ROUTER_ENABLE_API_KEY is enable.")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def log_configuration(args: argparse.Namespace) -> None:
    """
    Print all environment variables and CLI arguments prior to startup (with sanitization)
    """
    logger.info("=" * 80)
    logger.info("PD PROXY SERVER CONFIGURATION")
    logger.info("=" * 80)

    # Environment Variables
    logger.info("Environment Variables:")
    logger.info("  VLLM_PD_FIRST_TOKEN_FROM_P: %s", VLLM_PD_FIRST_TOKEN_FROM_P)
    logger.info("  VLLM_ROUTER_ENABLE_API_KEY: %s", VLLM_ROUTER_ENABLE_API_KEY)
    logger.info("  VLLM_ROUTER_ENABLE_DPLB: %s", VLLM_ROUTER_ENABLE_DPLB)
    logger.info("  VLLM_TRANSFER_ENABLE_LAYERWISE: %s", VLLM_TRANSFER_ENABLE_LAYERWISE)
    logger.info("  VLLM_ROUTER_DPLB_TOKENIZER_PATH: %s", VLLM_ROUTER_DPLB_TOKENIZER_PATH)
    logger.info("  OPENAI_API_KEY: %s", "SET" if os.environ.get("OPENAI_API_KEY") else "NOT SET")
    logger.info("")

    # Command-Line Arguments
    logger.info("Command-Line Arguments:")
    logger.info("  host: %s", args.host)
    logger.info("  port: %d", args.port)
    logger.info("  prefiller_instances: %d", len(args.prefiller_instances))
    for i, (host, port) in enumerate(args.prefiller_instances):
        logger.info("    [%d] %s:%d", i, host, port)
    logger.info("  decoder_instances: %d", len(args.decoder_instances))
    for i, (host, port) in enumerate(args.decoder_instances):
        logger.info("    [%d] %s:%d", i, host, port)

    # Sensitive Information Masking
    if args.api_keys:
        logger.info("  api_keys: %d key(s) configured", len(args.api_keys))
        for i, key in enumerate(args.api_keys):
            # Display only first 4 and last 4 characters for security
            masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
            logger.info("    [%d] %s", i, masked)
    else:
        logger.info("  api_keys: NONE (authentication disabled)")

    logger.info("=" * 80)


# Global variables
prefill_balancer = TokenLoadBalancer()
decode_balancer = TokenLoadBalancer()
client_session: aiohttp.ClientSession = None
token_manager: TokenManager = None
proxy_state = ProxyState()


class ProxyServer:
    def __init__(self, config: dict):
        self.config = config
        self._shutdown_event = asyncio.Event()
        self._site: Optional[web.TCPSite] = None

    async def run(self) -> None:
        """Main execution loop"""
        try:
            self._setup_resource_limits()

            self._init_token_manager()

            self._register_instances()

            await self._create_client_session()

            await self._start_http_server()

            # Await shutdown signal
            await self._shutdown_event.wait()

        except asyncio.CancelledError:
            logger.info("Server shutdown requested")
        finally:
            await self._cleanup()

    def shutdown(self) -> None:
        """Trigger graceful shutdown"""
        self._shutdown_event.set()

    def _setup_resource_limits(self):
        """Set file descriptor limits"""
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
        logger.info("Resource resource.RLIMIT_NOFILE: %s, hard_limit: %s", resource.RLIMIT_NOFILE, hard_limit)

    def _init_token_manager(self):
        """Initialize API key authentication"""
        global token_manager
        token_manager = TokenManager(self.config["api_keys"], enable_auth=VLLM_ROUTER_ENABLE_API_KEY)

    def _register_instances(self):
        """Statically register instances"""
        global prefill_balancer, decode_balancer
        for i, (host, port) in enumerate(self.config["prefiller_instances"]):
            id = f"prefiller_{i}"
            instance_p = LLMInstance(id, host, port)
            prefill_balancer.add(instance_p)

        for i, (host, port) in enumerate(self.config["decoder_instances"]):
            id = f"decoder_{i}"
            instance_d = LLMInstance(id, host, port)
            decode_balancer.add(instance_d)


    async def _create_client_session(self):
        """Create global ClientSession"""
        global client_session
        client_session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            connector=aiohttp.TCPConnector(limit=32768, limit_per_host=8192, force_close=True),
            timeout=aiohttp.ClientTimeout(total=0),
        )

    async def _start_http_server(self):
        """Start HTTP server"""
        app = web.Application(client_max_size=16 * 1024**2)
        app.add_routes(routes)

        runner = web.AppRunner(app)
        await runner.setup()

        self._site = web.TCPSite(runner, host=self.config["host"], port=self.config["port"], backlog=4096)
        await self._site.start()

        if VLLM_TRANSFER_ENABLE_LAYERWISE:
            global proxy_state
            await proxy_state.register_meta_server(self.config["host"], self.config["port"])

        logger.info(
            "Proxy server listening on http://%s:%s (Press CTRL+C to quit)", self.config["host"], self.config["port"]
        )

    async def _cleanup(self):
        """Clean up resources"""
        logger.info("Shutting down server...")

        global client_session
        if client_session:
            await client_session.close()

        if self._site:
            await self._site.stop()

        logger.info("Server shutdown complete")


def setup_signal_handlers(server: ProxyServer):
    """Set up signal handlers"""

    def signal_handler(signum, _):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        server.shutdown()

    # Register signal handlers (effective only in the main thread)
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill -15
    signal.signal(signal.SIGHUP, signal_handler)   # Configuration reload


async def main():
    """Main coroutine function"""

    config = parse_args()

    log_configuration(config)

    server = ProxyServer(vars(config))

    setup_signal_handlers(server)

    # Run server (blocks indefinitely until signal is triggered)
    await server.run()


if __name__ == "__main__":
    uvloop.run(main())
