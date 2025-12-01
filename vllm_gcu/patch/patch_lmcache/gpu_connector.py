#!/usr/bin/env python
# coding=utf-8
import torch
from unittest.mock import patch

try:
    from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2

    class VLLMGCUPagedMemGPUConnectorV2(VLLMPagedMemGPUConnectorV2):

        def _initialize_pointers(self, kv_caches) -> torch.Tensor:
            self.kv_cache_pointers.numpy()[:] = [
                t.data_ptr() for t in kv_caches
            ]
            device = kv_caches[0].device
            assert device.type == "gcu", "The device should be GCU."
            idx = device.index
            if idx not in self.kv_cache_pointers_on_gpu:
                self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                    self.num_layers, dtype=torch.int64, device=device)
            self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)
            if self.use_mla:
                # kv_caches[0].shape: [num_pages, page_size, head_size]
                assert kv_caches[0].dim() == 3
                self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[
                    0].shape[1]
            else:
                # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
                assert kv_caches[0].dim() == 5
                self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[
                    0].shape[2]

            return self.kv_cache_pointers_on_gpu[idx]

    patch("lmcache.v1.gpu_connector.VLLMPagedMemGPUConnectorV2",
          VLLMPagedMemGPUConnectorV2).start()
except Exception:
    pass
