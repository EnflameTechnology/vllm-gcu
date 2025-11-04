from unittest.mock import patch
from dataclasses import dataclass
from typing import Optional, cast

from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.sampling_params import SamplingType

_original_add_request = InputBatch.add_request
def _patched_add_request(self,
    request: "CachedRequestState",
) -> int:
    req_index = _original_add_request(self, request)
    if sampling_params := request.sampling_params:
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.temperature_cpu[req_index] = float('inf')
        else:
            self.temperature_cpu[req_index] = 1.0 / sampling_params.temperature
    return req_index

patch("vllm.v1.worker.gpu_input_batch.InputBatch.add_request", _patched_add_request).start()
