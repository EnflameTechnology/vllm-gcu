from vllm.v1.kv_offload.cpu import (
    AttentionLayerBase,
    CpuGpuOffloadingHandler,
    GPULoadStoreSpec,
    CPULoadStoreSpec,
    get_layers_from_vllm_config,
)
from unittest.mock import patch


def get_handlers(self, kv_caches):
    if not self._handler:

        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(self.vllm_config,
                                             AttentionLayerBase, layer_names)
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }

        self._handler = CpuGpuOffloadingHandler(
            attn_backends=attn_backends,
            gpu_block_size=self.gpu_block_size,
            cpu_block_size=self.offloaded_block_size,
            num_cpu_blocks=self.num_cpu_blocks,
            gpu_caches=kv_caches)

    assert self._handler is not None
    yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
    yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler


patch('vllm.v1.kv_offload.cpu.CPUOffloadingSpec.get_handlers',
      get_handlers).start()
