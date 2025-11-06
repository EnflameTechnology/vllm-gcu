import vllm.v1.engine.core as core
import vllm_gcu.envs as gcu_envs
from typing import Optional, Callable
from collections import deque
from unittest.mock import patch
from vllm.logger import init_logger
from vllm.v1.engine import (EngineCoreOutputs)
from vllm.version import __version__ as VLLM_VERSION
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.config import VllmConfig
from vllm.v1.executor.abstract import Executor
from concurrent.futures import Future



logger = init_logger(__name__)

def run_busy_loop(self):
    """Core busy loop of the EngineCore for data parallel case."""

    # Loop until process is sent a SIGINT or SIGTERM
    while True:
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()

        # 2) Step the engine core.
        executed, produced_output = self._process_engine_step()

        while not executed and produced_output:
            # NOTE(woosuk): This branch is taken when the previous step_fn
            # call updated the scheduler or worker states without actually
            # executing the model. With asynchronous scheduling, this
            # typically occurs every other step. To avoid unnecessary dummy
            # runs, we give step_fn a second chance to execute the model if
            # possible.
            executed, produced_output = self._process_engine_step()

        self._maybe_publish_request_counts()

        local_unfinished_reqs = self.scheduler.has_unfinished_requests()
        
        if not executed:
            if not local_unfinished_reqs and not self.engines_running:
                # All engines are idle.
                continue

            # We are in a running state and so must execute a dummy pass
            # if the model didn't execute any ready requests.
            self.execute_dummy_batch()

        # 3) All-reduce operation to determine global unfinished reqs.
        self.engines_running = self._has_global_unfinished_reqs(
            local_unfinished_reqs)

        if not self.engines_running:
            if self.dp_rank == 0 or not self.has_coordinator:
                # Notify client that we are pausing the loop.
                logger.debug("Wave %d finished, pausing engine loop.",
                                self.current_wave)
                # In the coordinator case, dp rank 0 sends updates to the
                # coordinator. Otherwise (offline spmd case), each rank
                # sends the update to its colocated front-end process.
                client_index = -1 if self.has_coordinator else 0
                self.output_queue.put_nowait(
                    (client_index,
                        EngineCoreOutputs(wave_complete=self.current_wave)))
            self.current_wave += 1


old_init = core.EngineCore.__init__
def __init__patch(self: core.EngineCore,
                vllm_config: VllmConfig,
                executor_class: type[Executor],
                log_stats: bool,
                executor_fail_callback: Optional[Callable] = None):
    old_init(self, vllm_config, executor_class, log_stats, executor_fail_callback)
    self.batch_queue: Optional[deque[tuple[Future[ModelRunnerOutput],
                                        SchedulerOutput]]] = None # type: ignore
    if self.batch_queue_size > 1:
        logger.info("Batch queue is enabled with size %d",
                    self.batch_queue_size)
        self.batch_queue = deque(maxlen=self.batch_queue_size)

def execute_model_with_error_logging(
    self: core.EngineCore,
    model_fn: Callable[[SchedulerOutput], ModelRunnerOutput],
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    """Execute the model and log detailed info on failure."""
    try:
        return model_fn(scheduler_output)
    except Exception as err:
        # We do not want to catch BaseException here since we're only
        # interested in dumping info when the exception is due to an
        # error from execute_model itself.

        # NOTE: This method is exception-free
        raise err

old_step_witch_batch = core.EngineCore.step_with_batch_queue
def step_with_batch_queue_patch(
        self: core.EngineCore) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
    """Schedule and execute batches with the batch queue.
    Note that if nothing to output in this step, None is returned.

    The execution flow is as follows:
    1. Try to schedule a new batch if the batch queue is not full.
    If a new batch is scheduled, directly return an empty engine core
    output. In other words, fulfilling the batch queue has a higher priority
    than getting model outputs.
    2. If there is no new scheduled batch, meaning that the batch queue
    is full or no other requests can be scheduled, we block until the first
    batch in the job queue is finished.
    3. Update the scheduler from the output.
    """
    if not self.vllm_config.additional_config.get("async_scheduling", False):
        return old_step_witch_batch(self)
    batch_queue = self.batch_queue
    assert batch_queue is not None

    # Try to schedule a new batch if the batch queue is not full, but
    # the scheduler may return an empty batch if all requests are scheduled.
    # Note that this is not blocking.
    assert len(batch_queue) < self.batch_queue_size

    model_executed = False
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output)
        model_executed = scheduler_output.total_num_scheduled_tokens > 0
        batch_queue.appendleft(
            (future, scheduler_output))  # type: ignore[arg-type]

        if model_executed and len(batch_queue) < self.batch_queue_size \
            and not batch_queue[-1][0].done():
            # Don't block on next worker response unless the queue is full
            # or there are no more requests to schedule.
            return None, True
    elif not batch_queue:
        # Queue is empty. We should not reach here since this method should
        # only be called when the scheduler contains requests or the queue
        # is non-empty.
        return None, False

    # Block until the next result is available.
    future, scheduler_output = batch_queue.pop()
    model_output = self.execute_model_with_error_logging(
        lambda _: future.result(), scheduler_output)

    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)

    return engine_core_outputs, model_executed

def execute_dummy_batch_patch(self) -> None:
    enable_async_scheduling = self.vllm_config.additional_config.get("async_scheduling", False)
    self.model_executor.collective_rpc("execute_dummy_batch", non_block=enable_async_scheduling)

patch.object(core.EngineCore, "__init__", __init__patch).start()
patch.object(core.EngineCore, "execute_model_with_error_logging", execute_model_with_error_logging, create=True).start()
patch.object(core.EngineCore, "step_with_batch_queue", step_with_batch_queue_patch).start()
patch.object(core.EngineCore, "execute_dummy_batch", execute_dummy_batch_patch).start()