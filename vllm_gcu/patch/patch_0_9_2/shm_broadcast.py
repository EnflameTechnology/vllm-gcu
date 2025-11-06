import asyncio
import pickle
import time
import vllm.distributed.device_communicators.shm_broadcast as shm
import vllm.envs as envs
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Optional, Union
from threading import Event
from unittest.mock import patch
from vllm.logger import init_logger

logger = init_logger(__name__)

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL

@asynccontextmanager
async def acquire_read_async(self,
                    timeout: Optional[float] = None,
                    cancel: Optional[Event] = None):
    assert self._is_local_reader, "Only readers can acquire read"
    start_time = time.monotonic()
    n_warning = 1
    while True:
        with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
            read_flag = metadata_buffer[self.local_reader_rank + 1]
            written_flag = metadata_buffer[0]
            if not written_flag or read_flag:
                # this block is either
                # (1) not written
                # (2) already read by this reader

                # for readers, `self.current_idx` is the next block to read
                # if this block is not ready,
                # we need to wait until it is written

                # Release the processor to other threads
                self._read_spin_timer.spin()

                # if we wait for a long time, log a message
                if (time.monotonic() - start_time
                        > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning):
                    logger.debug(
                        ("No available shared memory broadcast block found"
                            " in %s second."),
                        VLLM_RINGBUFFER_WARNING_INTERVAL,
                    )
                    n_warning += 1

                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")

                # if we time out, raise an exception
                if (timeout is not None
                        and time.monotonic() - start_time > timeout):
                    raise TimeoutError
                await asyncio.sleep(0)
                continue
            # found a block that is not read by this reader
            # let caller read from the buffer
            with self.buffer.get_data(self.current_idx) as buf:
                yield buf

            # caller has read from the buffer
            # set the read flag
            metadata_buffer[self.local_reader_rank + 1] = 1
            self.current_idx = (self.current_idx +
                                1) % self.buffer.max_chunks

            self._read_spin_timer.record_activity()
            break

async def dequeue_async(self,
            timeout: Optional[float] = None,
            cancel: Optional[Event] = None):
    """ Read from message queue with optional timeout (in seconds) """
    if self._is_local_reader:
        async with self.acquire_read_async(timeout, cancel) as buf:
            overflow = buf[0] == 1
            if not overflow:
                # no need to know the size of serialized object
                # pickle format contains the size information internally
                # see https://docs.python.org/3/library/pickle.html
                obj = pickle.loads(buf[1:])
        if overflow:
            obj = shm.MessageQueue.recv(self.local_socket, timeout)
    elif self._is_remote_reader:
        obj = shm.MessageQueue.recv(self.remote_socket, timeout)
    else:
        raise RuntimeError("Only readers can dequeue")
    return obj

async def __aenter__(self):
    return self

async def __aexit__(self):
    pass

patch.object(shm.MessageQueue, "acquire_read_async", acquire_read_async, create=True).start()
patch.object(shm.MessageQueue, "dequeue_async", dequeue_async, create=True).start()
patch.object(shm.MessageQueue, "__aenter__", __aenter__, create=True).start()
patch.object(shm.MessageQueue, "__aexit__", __aexit__, create=True).start()
