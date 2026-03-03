"""
Background async worker for EPLB weight transfer.

Adapted from vLLM v0.14.1's async_worker.py. The worker runs in a daemon
thread and transfers expert weights one layer at a time into a staging
buffer. The main inference thread polls for readiness and copies the
buffer into the model weights between forward passes.
"""

import asyncio
import threading
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger

from .rebalance_execute import transfer_layer_gcu

if TYPE_CHECKING:
    from .async_state import AsyncEplbExtension

logger = init_logger(__name__)


def start_async_worker(
    eplb_state,
    async_ext: "AsyncEplbExtension",
    rank_mapping: Optional[Dict[int, int]] = None,
    is_profile: bool = False,
) -> threading.Thread:
    """Start the daemon thread that drives async EPLB layer transfers."""
    ep_group = get_ep_group().device_group
    rank = ep_group.rank()
    device_index = async_ext.cuda_device_index

    def thread_target() -> None:
        assert device_index is not None
        torch.cuda.set_device(device_index)
        cuda_stream = torch.cuda.Stream(device=device_index)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _transfer_run_periodically(
                    eplb_state=eplb_state,
                    async_ext=async_ext,
                    ep_group=ep_group,
                    is_profile=is_profile,
                    rank_mapping=rank_mapping,
                    cuda_stream=cuda_stream,
                )
            )
        except Exception as exc:
            logger.exception(
                "async EPLB loop error (rank %d): %s", rank, str(exc)
            )
        finally:
            loop.close()

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


async def _transfer_run_periodically(
    eplb_state,
    async_ext: "AsyncEplbExtension",
    ep_group: ProcessGroup,
    is_profile: bool = False,
    rank_mapping: Optional[Dict[int, int]] = None,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """
    Event loop that waits for rearrange signals and transfers layers
    one at a time into the staging buffer.
    """
    while True:
        await asyncio.to_thread(async_ext.rearrange_event.wait)
        logger.info("async EPLB worker woke up for transfer")

        model = async_ext.model
        num_moe_layers = model.num_moe_layers

        while (
            async_ext.rebalanced
            and async_ext.layer_to_transfer < num_moe_layers
        ):
            if (
                not async_ext.ep_buffer_ready
                and async_ext.rebalanced
                and async_ext.new_physical_to_logical_map is not None
            ):
                await asyncio.to_thread(async_ext.buffer_lock.acquire)
                try:
                    if async_ext.layer_to_transfer >= num_moe_layers:
                        break

                    if async_ext.buffer_consumed_event is not None:
                        cuda_stream.wait_event(
                            async_ext.buffer_consumed_event)
                        async_ext.buffer_consumed_event = None

                    (
                        async_ext.is_unchanged,
                        async_ext.is_received_locally,
                        async_ext.recv_metadata,
                    ) = await transfer_layer_gcu(
                        old_global_expert_indices=eplb_state.physical_to_logical_map,
                        new_global_expert_indices=async_ext.new_physical_to_logical_map,
                        expert_weights=model.expert_weights,
                        expert_weights_buffer=async_ext.expert_buffer,
                        ep_group=ep_group,
                        is_profile=is_profile,
                        layer=async_ext.layer_to_transfer,
                        cuda_stream=cuda_stream,
                        rank_mapping=rank_mapping,
                    )
                    event = torch.cuda.Event(blocking=False)
                    if cuda_stream is not None:
                        cuda_stream.record_event(event)
                    async_ext.buffer_ready_event = event
                finally:
                    async_ext.buffer_lock.release()
                async_ext.ep_buffer_ready = 1
            else:
                if not async_ext.rebalanced:
                    break
                await asyncio.sleep(0.001)

        async_ext.rearrange_event.clear()
