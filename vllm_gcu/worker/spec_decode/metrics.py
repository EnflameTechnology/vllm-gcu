from typing import Callable, Optional, Union

import torch
import torch_gcu

from vllm.spec_decode.metrics import AsyncMetricsCollector, SpecDecodeWorkerMetrics

class AsyncMetricsGCUCollector(AsyncMetricsCollector):

    def init_gpu_tensors(self, rank: int) -> None:
        self._rank = rank
        self._copy_stream = torch.gcu.Stream()

    def init_tensors(self,
                     rank: int,
                     device_type: Union[torch.device, str] = 'gcu') -> None:
        self._rank = rank
        if isinstance(device_type, torch.device):
            device_type = device_type.type
        if device_type == 'gcu':
            self._copy_stream = torch.gcu.Stream()

    def maybe_collect_rejsample_metrics(
            self, k: int) -> Optional[SpecDecodeWorkerMetrics]:

        # If a copy was initiated in the previous call, collect and return.
        if self._in_flight_copy is not None:
            ready_event = self._in_flight_copy
            self._in_flight_copy = None
            return self._collect_rejsample_metrics(k, ready_event)

        # Otherwise, check if we should start a new copy.
        if self._should_collect_rejsample_metrics(self._timer()):
            assert self._in_flight_copy is None
            self._in_flight_copy = self._copy_rejsample_metrics_async()

        return None

    def _copy_rejsample_metrics_async(self) -> torch.gcu.Event:
        """Copy rejection/typical-acceptance sampling metrics
        (number of accepted tokens, etc) to CPU asynchronously.

        Returns a CUDA event recording when the copy is complete.
        """
        assert self._copy_stream is not None
        self._copy_stream.wait_stream(torch.gcu.current_stream())

        with torch.gcu.stream(self._copy_stream):
            self._aggregate_num_accepted_tokens.copy_(
                self.spec_decode_sampler.num_accepted_tokens,
                non_blocking=True)
            self._aggregate_num_emitted_tokens.copy_(
                self.spec_decode_sampler.num_emitted_tokens, non_blocking=True)
            # Number of draft tokens is calculated on CPU, so no copy is
            # required.
            self._aggregate_num_draft_tokens = (
                self.spec_decode_sampler.num_draft_tokens)

        aggregate_metrics_ready = torch.gcu.Event()
        aggregate_metrics_ready.record(self._copy_stream)

        return aggregate_metrics_ready
