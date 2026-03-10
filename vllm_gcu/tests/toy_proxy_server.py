# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import argparse
import itertools
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List
import httpx
import resource
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.datastructures import Headers
import json
logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(
    level=logging.ERROR,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s'
)
# Environment variable to control whether to return first chunk from prefill
# If set to "1", "true", or "yes", the first chunk from prefill will be returned
# Default: False (all chunks come from decode)
VLLM_GCU_PD_FIRST_TOKEN_FROM_P = os.environ.get("VLLM_GCU_PD_FIRST_TOKEN_FROM_P", "0").lower() in ("1", "true", "yes")
# Environment variable to control whether to enable API key authentication
# If set to "true", "yes", or "1", API key authentication will be enabled
# Default: False (no authentication required)
VLLM_GCU_ROUTER_ENABLE_API_KEY = os.environ.get("VLLM_GCU_ROUTER_ENABLE_API_KEY", "false").lower() in ("1", "true", "yes")
class TokenManager:
    def __init__(self, api_keys: Optional[List[str]] = None, enable_auth: bool = False):
        """
        Initialize token manager with API keys.
        Args:
            api_keys: List of API keys. If None or empty, verification is disabled.
            enable_auth: Whether to enable authentication. Controlled by VLLM_GCU_ROUTER_ENABLE_API_KEY.
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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.token_manager = TokenManager(global_args.api_keys, enable_auth=VLLM_GCU_ROUTER_ENABLE_API_KEY)
    if app.state.token_manager.enabled:
        print(f"API key authentication ENABLED with {len(app.state.token_manager.api_tokens)} key(s)")
    else:
        print("WARNING: API key authentication is DISABLED")
    limits = httpx.Limits(
        max_connections=2048,
        max_keepalive_connections=2048,
        keepalive_expiry=1200
    )
    timeout = httpx.Timeout(
        connect=600,
        read=1200,
        write=1200,
        pool=1200
    )
    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    # http2=True,
                    timeout=timeout,
                    base_url=prefiller_base_url,
                    limits=limits,
                    verify=False,
                    # http1=False,
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )
    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    # http2=True,
                    timeout=timeout,
                    base_url=decoder_base_url,
                    limits=limits,
                    verify=False,
                    # http1=False,
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )
    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )
    yield
    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()
# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--api-keys",
                        "--api-key",
                        type=str,
                        nargs="+",
                        default=None,
                        help="API keys for authentication. If not provided, no authentication is required.")
    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"]
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )
    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )
    args = parser.parse_args()
    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args
async def verify_api_key(request: Request):
    """
    Dependency to verify API key from request headers.
    Raises HTTPException if verification fails.
    """
    token_manager = request.app.state.token_manager
    if not token_manager.verify_token(request.headers):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.
    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'
    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")
class PrefillServiceError(RuntimeError):
    """Raised when prefill returns non-2xx or network fails."""
    def __init__(self, endpoint, status, request_id, body):
        super().__init__(
            f"prefill {endpoint} returned {status}, req-id={request_id}, body={body}"
        )
        self.endpoint = endpoint
        self.status = status
        self.request_id = request_id
        self.body = body
class DecoderServiceError(RuntimeError):
    """Raised when decoder returns non-2xx or network fails."""
    def __init__(self, endpoint, status, request_id, body):
        super().__init__(
            f"decoder {endpoint} returned {status}, req-id={request_id}, body={body}"
        )
        self.endpoint = endpoint
        self.status = status
        self.request_id = request_id
        self.body = body
async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()
    return response
    # try:
    #     async with client_info["client"].post(
    #         endpoint,
    #         json=req_data,
    #         headers=headers,
    #         timeout=300.0,
    #     ) as resp:
    #         body = await resp.text()
    #         if resp.status >= 400:
    #             raise PrefillServiceError(endpoint, resp.status, request_id, body)
    #         return await resp.json()
    # except Exception as exc:
    #     if not isinstance(exc, PrefillServiceError):
    #         raise PrefillServiceError(endpoint, "network", request_id, str(exc)) from exc
    #     raise
async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk
async def send_request_to_decoder_service(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
) -> dict:
    """
    Send a **non-streaming** request to decoder and return JSON.
    Raises DecoderServiceError on any failure.
    """
    req_data = req_data.copy()
    req_data["stream"] = False
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()
    return response
    # try:
    #     async with client_info["client"].post(
    #         endpoint,
    #         json=req_data,
    #         headers=headers,
    #         timeout=timeout,
    #     ) as resp:
    #         body = await resp.text()
    #         if resp.status >= 400:
    #             raise DecoderServiceError(endpoint, resp.status, request_id, body)
    #         return await resp.json()
    # except Exception as exc:
    #     if not isinstance(exc, DecoderServiceError):
    #         raise DecoderServiceError(endpoint, "network", request_id, str(exc)) from exc
    #     raise
async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        # request_id = str(uuid.uuid4())
        request_id = req_data.get("query_id", str(uuid.uuid4()))
        # Get the next prefill client in round-robin fashion
        logger.info("Send to prefiller, req id: %s", request_id)
        prefill_client_info = get_next_client(request.app, "prefill")
        # Send request to prefill service
        # req_data["model"] = "/home/pretrained_models/deepseek-r1"
        response = await send_request_to_service(
            prefill_client_info, api, req_data, request_id
        )
        logger.info("Recv response from prefiller, req id: %s", request_id)
        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")
        # logger.debug("Using %s %s", prefill_client_info, decode_client_info)
        logger.info("Send req_data to decoder, req id: %s", request_id)
        from urllib.parse import urlparse
        def is_chat_api(path: str) -> bool:
            if not path:
                return False
            normalized = urlparse(path).path.rstrip("/")
            return normalized == "/v1/chat/completions"
        is_chat = is_chat_api(request.url.path) and request.method == "POST"
        is_stream = req_data.get("stream", False)
        logger.info(f"is_chat: {is_chat}, is_stream: {is_stream}")
        if not is_stream:
            logger.info("Non-streaming response from decoder, req id: %s", request_id)
            decode_response = await send_request_to_decoder_service(
                decode_client_info, api, req_data, request_id=request_id
            )
            decode_response_json = decode_response.json()
            if (VLLM_GCU_PD_FIRST_TOKEN_FROM_P):
                if response_json is None:
                    logger.error("Prefill response not found while first token needed")
                    raise RuntimeError("first token required but prefill missing")
                if is_chat:
                    first_token = response_json["choices"][0]["message"]["content"]
                    if first_token:
                        decode_response_json["choices"][0]["message"]["content"] = \
                            first_token + decode_response_json["choices"][0]["message"]["content"]
                    first_reasoning_token = response_json["choices"][0]["message"]["reasoning_content"]
                    if first_reasoning_token:
                        decode_response_json["choices"][0]["message"]["reasoning_content"] = \
                            first_reasoning_token + decode_response_json["choices"][0]["message"]["reasoning_content"]
                else:
                    first_token = response_json["choices"][0]["text"]
                    decode_response_json["choices"][0]["text"] = \
                        first_token + decode_response_json["choices"][0]["text"]
            logger.info("Recv response from decoder, req id: %s", request_id)
            return JSONResponse(content=decode_response_json)
        # Stream response from decode service
        async def generate_stream():
            # Only return first chunk from prefill if VLLM_GCU_PD_FIRST_TOKEN_FROM_P is enabled
            if VLLM_GCU_PD_FIRST_TOKEN_FROM_P:
                if is_chat:
                    # For chat completions, use delta format
                    first_chunk = {
                        "id": response_json["id"],
                        "object": "chat.completion.chunk",
                        "created": response_json["created"],
                        "model": response_json["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": response_json["choices"][0]["message"]["content"],
                                "reasoning_content": response_json["choices"][0]["message"]["reasoning_content"]
                            },
                            "finish_reason": None
                        }]
                    }
                else:
                    # For regular completions, use text format
                    first_chunk = {
                        "id": response_json["id"],
                        "object": "text_completion",
                        "created": response_json["created"],
                        "model": response_json["model"],
                        "choices": [{
                            "index": 0,
                            "text": response_json["choices"][0]["text"],
                            "logprobs": response_json["choices"][0].get("logprobs"),
                            "finish_reason": None
                        }]
                    }
                yield (
                    "data: " + json.dumps(first_chunk, separators=(",", ":"), ensure_ascii=False) + "\n\n"
                ).encode()
            first_chunk = True
            async for chunk in stream_service_response(
                decode_client_info, api, req_data, request_id=request_id
            ):
                # Skip first chunk from decode only if we already returned prefill first chunk
                if first_chunk and VLLM_GCU_PD_FIRST_TOKEN_FROM_P:
                    first_chunk = False
                    continue
                first_chunk = False
                # logger.info("Recv chunk from decoder, req id: %s", request_id)
                yield chunk
        # Dynamically set media_type based on stream parameter
        is_stream = req_data.get("stream", False)
        media_type = "text/event-stream" if is_stream else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
@app.post("/v1/completions")
async def handle_completions(request: Request, _: None = Depends(verify_api_key)):
    return await _handle_completions("/completions", request)
@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request, _: None = Depends(verify_api_key)):
    return await _handle_completions("/chat/completions", request)
@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }
if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    print(f"Starting server with host={global_args.host}, port={global_args.port}")
    print(f"Prefiller instances: {global_args.prefiller_instances}")
    print(f"Decoder instances: {global_args.decoder_instances}")
    print(f"API keys configured: {len(global_args.api_keys) if global_args.api_keys else 0}")
    print(f"VLLM_GCU_ROUTER_ENABLE_API_KEY: {VLLM_GCU_ROUTER_ENABLE_API_KEY}")
    import uvicorn
    uvicorn.run(
        app,
        host=global_args.host,
        port=global_args.port,
        loop="uvloop",
        http="httptools",
        backlog=2048,
        interface="asgi3",
        timeout_keep_alive=300,
    )
