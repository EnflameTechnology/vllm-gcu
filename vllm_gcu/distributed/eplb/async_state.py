"""
Async EPLB extension state.

This module provides AsyncEplbExtension which holds all async-specific state
for the EPLB feature ported from vLLM v0.14.1. It is designed to be attached
to the v0.11.0 EplbState dataclass instance at runtime, keeping the async
functionality fully decoupled from the base EPLB implementation.
"""

import threading
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from vllm.model_executor.models.interfaces import MixtureOfExperts

if TYPE_CHECKING:
    from .rebalance_execute import RecvMetadata


class AsyncEplbExtension:
    """
    Holds async-specific state for EPLB on top of v0.11.0 EplbState.

    When VLLM_GCU_EPLB_ASYNC_ENABLED=1, an instance of this class is lazily
    attached to the EplbState dataclass as `_async_ext`. It manages the
    background thread, transfer buffers, and layer-by-layer progress.
    """

    def __init__(self, model: MixtureOfExperts, device: torch.device):
        self.model = model
        self.device = device

        self._init_buffers(model)

        self.buffer_lock = threading.Lock()
        self.buffer_ready_event: Optional[torch.cuda.Event] = None
        self.buffer_consumed_event: Optional[torch.cuda.Event] = None
        self.ep_buffer_ready: int = 0
        self.layer_to_transfer: int = 0
        self.rebalanced: bool = False
        self.pending_global_ready_check: bool = False

        self.is_unchanged: np.ndarray = np.array([], dtype=np.bool_)
        self.is_received_locally: np.ndarray = np.array([], dtype=np.bool_)
        self.recv_metadata: Optional["RecvMetadata"] = None

        self.new_physical_to_logical_map: Optional[torch.Tensor] = None
        self.new_logical_to_physical_map: Optional[torch.Tensor] = None
        self.new_logical_replica_count: Optional[torch.Tensor] = None

        self.rearrange_event = threading.Event()
        self.async_worker: Optional[threading.Thread] = None

        self.cuda_device_index: Optional[int] = None
        if device.type == "gcu":
            self.cuda_device_index = device.index
            if self.cuda_device_index is None and torch.cuda.is_available():
                self.cuda_device_index = torch.cuda.current_device()

    def _init_buffers(self, model: MixtureOfExperts):
        """Initialize per-weight buffers for expert weight transfer (v0.14.1 style)."""
        self.expert_buffer: list[torch.Tensor] = [
            torch.empty_like(w) for w in model.expert_weights[0]
        ]
