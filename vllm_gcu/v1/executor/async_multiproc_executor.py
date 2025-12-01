# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import cloudpickle
import signal
import traceback
import queue
import vllm_gcu.patch
import vllm_gcu.envs as gcu_envs

from enum import Enum, auto
from functools import partial
from typing import Tuple
from collections import deque
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import Handle
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor,  WorkerProc
from unittest.mock import patch


logger = init_logger(__name__)

class AsyncWorkerProc(WorkerProc):
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
    ):
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, input_shm_handle)
        self.model_output_queue = queue.Queue()
        self.output_queue = deque()
        additional_config = vllm_config.additional_config
        self.enable_async_executing = additional_config.get("async_executing", False)
        self.enable_async_scheduling = additional_config.get("async_scheduling", False)

    async def handle_output_queue(self):
        while True:
            if len(self.output_queue) == 0:
                await asyncio.sleep(0)
                continue
            worker_output = self.output_queue[0]
            if not isinstance(worker_output, asyncio.Task):
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.SUCCESS, worker_output))

                self.output_queue.popleft()
                continue
            
            model_output = await worker_output

            self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.SUCCESS, model_output))

            self.output_queue.popleft()
            
    async def worker_busy_loop_async(self):
        """Main busy loop for Multiprocessing Workers"""

        asyncio.create_task(self.handle_output_queue())

        while True:
            method, args, kwargs, output_rank = await self.rpc_broadcast_mq.dequeue_async()
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)

                if isinstance(output, Tuple):
                    coroutine, is_driver_worker = output
                    # 创建任务
                    task = asyncio.create_task(coroutine)
                    if is_driver_worker:
                        self.output_queue.append(task)
                    
                    # 任务创建完之后马上放弃CPU执行权限
                    await asyncio.sleep(0)

                    continue
                else:
                    if output_rank is None or self.rank == output_rank:
                        self.output_queue.append(output)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.FAILURE, str(e)))
                continue
            
    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        try:
            reader.close()
            worker = AsyncWorkerProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_writer.send({
                "status":
                WorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
            })

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None
            
            if worker.enable_async_executing:
                asyncio.run(worker.worker_busy_loop_async())
            else:
                worker.worker_busy_loop()

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True
        finally:
            if ready_writer is not None:
                ready_writer.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()


    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()


class AsyncMultiprocExecutor(MultiprocExecutor):

    def _init_executor(self) -> None:
        import vllm_gcu.patch
        with patch("vllm.v1.executor.multiproc_executor.WorkerProc",AsyncWorkerProc):
            super()._init_executor()

    @property
    def max_concurrent_batches(self) -> int:
        async_scheduling = self.vllm_config.additional_config.get('async_scheduling', False)
        if async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size
