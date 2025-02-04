#!/usr/bin/env python
# coding=utf-8

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

logger = init_logger(__name__)


class GCUBaseKVCacheMethod(BaseKVCacheMethod):
    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.k_zero = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        layer.v_zero = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
            if layer.k_scale > 0.0 and layer.v_scale > 0.0:
                k_scale = layer.k_scale.to("cpu").tolist()
                v_scale = layer.v_scale.to("cpu").tolist()
                k_zero = layer.k_zero.to("cpu").tolist()
                v_zero = layer.v_zero.to("cpu").tolist()
            elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
                k_scale = 1.0
                v_scale = 1.0
                k_zero = 0.0
                v_zero = 0.0
            else:
                assert layer.k_scale > 0.0
                scale_to_duplicate = max(layer.k_scale, layer.v_scale)
                zero_to_duplicate = max(layer.k_zero, layer.v_zero)
                k_scale = scale_to_duplicate.to("cpu").tolist()
                v_scale = scale_to_duplicate.to("cpu").tolist()
                k_zero = zero_to_duplicate.to("cpu").tolist()
                v_zero = zero_to_duplicate.to("cpu").tolist()

            if not isinstance(k_scale, float) or not isinstance(v_scale, float):
                raise ValueError(
                    "Only support per-tensor scaling factor for int8 KV cache"
                )

            layer._k_scale.copy_(k_scale)
            layer._v_scale.copy_(v_scale)
            layer._k_scale_float = k_scale
            layer._v_scale_float = v_scale
            layer._k_zero_float = k_zero
            layer._v_zero_float = v_zero

        del layer.k_scale
        del layer.v_scale
        del layer.k_zero
        del layer.v_zero
