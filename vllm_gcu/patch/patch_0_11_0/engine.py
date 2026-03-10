from unittest.mock import patch

from typing import Any, Callable, Optional, TypeVar, Union
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient, DPAsyncMPClient
from vllm.v1.engine.processor import Processor
from vllm_gcu.utils import get_tx_ctx, get_tx_mark_func
from vllm.utils import (decorate_logs, get_hash_fn_by_name, make_zmq_socket,
                        resolve_obj_by_qualname, set_process_title)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
import vllm_gcu.envs as gcu_envs
import vllm.envs as envs
from vllm.v1.engine.core import EngineCoreProc

from collections import deque
from contextlib import ExitStack, contextmanager
import zmq
import orjson
import time


def post_step(self, model_executed: bool) -> None:
    use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling
    if not use_async_scheduling and self.use_spec_decode and model_executed:
        # Take the draft token ids.
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)


# Save the original function reference
_original_preprocess_add_request = EngineCore.preprocess_add_request


def preprocess_add_request(self, request):
    message = "preprocess_add_request"
    color = "blue"
    domain = "VLLM"
    category = f"EngineCore-DP{self.engine_index}"
    payload = {
        "req_ids": [request.request_id]
    }
    payload_str = orjson.dumps(payload)

    with get_tx_ctx(message, color, domain, category, payload_str):
        result = _original_preprocess_add_request(self, request)

    return result


add_request_async_async_mp_client = AsyncMPClient.add_request_async
add_request_async_dp_async_mp_client = DPAsyncMPClient.add_request_async

async def add_request_async(self, request) -> None:
    message = "add_request_async"
    color = "blue"
    domain = "VLLM"
    category = "EngineCoreClient"
    payload = {
        "req_ids": [request.request_id]
    }
    payload_str = orjson.dumps(payload)


    with get_tx_ctx(message, color, domain, category, payload_str):
        await add_request_async_async_mp_client(self, request)


async def add_request_async_dp(self, request) -> None:
    message = "add_request_async"
    color = "blue"
    domain = "VLLM"
    category = "EngineCoreClient"
    payload = {
        "req_ids": [request.request_id]
    }
    payload_str = orjson.dumps(payload)

    with get_tx_ctx(message, color, domain, category, payload_str):
        await add_request_async_dp_async_mp_client(self, request)


origin_process_inputs = Processor.process_inputs

def process_inputs(
    self,
    request_id,
    prompt,
    params,
    arrival_time = None,
    lora_request = None,
    tokenization_kwargs = None,
    trace_headers = None,
    priority = 0,
    data_parallel_rank = None,
):
    message = "arrive"
    color = "blue"
    domain = "VLLM"
    category = "AsyncLLM"
    payload = {
        "req_ids": [request_id]
    }
    payload_str = orjson.dumps(payload)

    tx_mark_func = get_tx_mark_func()
    tx_mark_func(message, color, domain, category, payload_str)

    return origin_process_inputs(self,
                                 request_id,
                                 prompt, params,
                                 arrival_time,
                                 lora_request,
                                 tokenization_kwargs,
                                 trace_headers,
                                 priority,
                                 data_parallel_rank)


def process_output_sockets(self, output_paths: list[str],
                            coord_output_path: Optional[str],
                            engine_index: int):
    """Output socket IO thread."""

    # Msgpack serialization encoding.
    encoder = MsgpackEncoder()
    # Send buffers to reuse.
    reuse_buffers: list[bytearray] = []
    # Keep references to outputs and buffers until zmq is finished
    # with them (outputs may contain tensors/np arrays whose
    # backing buffers were extracted for zero-copy send).
    pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

    # We must set linger to ensure the ENGINE_CORE_DEAD
    # message is sent prior to closing the socket.
    with ExitStack() as stack, zmq.Context() as ctx:
        sockets = [
            stack.enter_context(
                make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
            for output_path in output_paths
        ]
        coord_socket = stack.enter_context(
            make_zmq_socket(
                ctx, coord_output_path, zmq.PUSH, bind=False,
                linger=4000)) if coord_output_path is not None else None
        max_reuse_bufs = len(sockets) + 1

        while True:
            output = self.output_queue.get()
            if output == EngineCoreProc.ENGINE_CORE_DEAD:
                for socket in sockets:
                    socket.send(output)
                break
            assert not isinstance(output, bytes)
            client_index, outputs = output
            outputs.engine_index = engine_index

            if client_index == -1:
                # Don't reuse buffer for coordinator message
                # which will be very small.
                assert coord_socket is not None
                coord_socket.send_multipart(encoder.encode(outputs))
                continue

            # Reclaim buffers that zmq is finished with.
            while pending and pending[-1][0].done:
                reuse_buffers.append(pending.pop()[2])

            buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
            buffers = encoder.encode_into(outputs, buffer)

            # trace start
            message = "process_output_sockets"
            color = "blue"
            domain = "VLLM"
            category = f"EngineCore-DP{self.engine_index}"
            payload = {"req_ids": []}

            for output in outputs.outputs:
                request_id = output.request_id
                payload["req_ids"].append(request_id)
            payload_str = orjson.dumps(payload)

            tx_mark_func = get_tx_mark_func()
            tx_mark_func(message, color, domain, category, payload_str)
            # trace end

            tracker = sockets[client_index].send_multipart(buffers,
                                                            copy=False,
                                                            track=True)
            if not tracker.done:
                ref = outputs if len(buffers) > 1 else None
                pending.appendleft((tracker, ref, buffer))
            elif len(reuse_buffers) < max_reuse_bufs:
                # Limit the number of buffers to reuse.
                reuse_buffers.append(buffer)

origin_get_output_async = AsyncMPClient.get_output_async

async def get_output_async(self):
    outputs = await origin_get_output_async(self)

    # trace start
    message = "get_output_async"
    color = "blue"
    domain = "VLLM"
    category = "EngineCoreClient"
    payload = {"req_ids": []}

    for output in outputs.outputs:
        request_id = output.request_id
        payload["req_ids"].append(request_id)
    payload_str = orjson.dumps(payload)

    tx_mark_func = get_tx_mark_func()
    tx_mark_func(message, color, domain, category, payload_str)
    # trace end

    return outputs


patch("vllm.v1.engine.core.EngineCore.post_step", post_step).start()

if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
    patch("vllm.v1.engine.core.EngineCore.preprocess_add_request",
          preprocess_add_request).start()
    patch("vllm.v1.engine.core_client.AsyncMPClient.add_request_async",
          add_request_async).start()
    patch("vllm.v1.engine.core_client.DPAsyncMPClient.add_request_async",
          add_request_async_dp).start()
    patch("vllm.v1.engine.processor.Processor.process_inputs",
          process_inputs).start()
    patch("vllm.v1.engine.core.EngineCoreProc.process_output_sockets",
          process_output_sockets).start()
    patch("vllm.v1.engine.core_client.AsyncMPClient.get_output_async",
          get_output_async).start()