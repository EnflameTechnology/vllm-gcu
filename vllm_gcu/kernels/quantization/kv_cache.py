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


class GCUInt8KVCachePerTensorMethod(GCUBaseKVCacheMethod):
    """
    Int8 KV cache quantization method with per-tensor scaling.

    Supports both symmetric and asymmetric quantization:
    - Symmetric quantization: k_zero/v_zero are not provided in checkpoint,
      layer._k_zero/layer._v_zero will be set to None
    - Asymmetric quantization: k_zero/v_zero are provided in checkpoint,
      layer._k_zero/layer._v_zero will be set to the loaded values
    """

    # Magic number to indicate uninitialized zero point (using NaN)
    _ZERO_POINT_MAGIC = float('nan')

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        # Initialize scale with -1.0 (invalid value, will be overwritten if provided)
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        # Initialize zero point with NaN as magic number
        # If not overwritten by checkpoint, it means symmetric quantization
        layer.k_zero = torch.nn.Parameter(
            torch.tensor(self._ZERO_POINT_MAGIC), requires_grad=False
        )
        layer.v_zero = torch.nn.Parameter(
            torch.tensor(self._ZERO_POINT_MAGIC), requires_grad=False
        )

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def _is_symmetric_quant(self, zero_tensor: torch.Tensor) -> bool:
        """Check if zero point is still the magic number (not loaded from checkpoint)."""
        return torch.isnan(zero_tensor).item()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Determine if this is symmetric or asymmetric quantization
        # by checking if k_zero/v_zero were loaded (not NaN anymore)
        is_symmetric = (self._is_symmetric_quant(layer.k_zero) and
                        self._is_symmetric_quant(layer.v_zero))

        if is_symmetric:
            logger.debug("Detected symmetric int8 KV cache quantization "
                        "(k_zero/v_zero not provided in checkpoint)")
        else:
            logger.debug("Detected asymmetric int8 KV cache quantization "
                        "(k_zero/v_zero loaded from checkpoint)")

        if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
            if layer.k_scale > 0.0 and layer.v_scale > 0.0:
                k_scale = layer.k_scale.to("cpu").tolist()
                v_scale = layer.v_scale.to("cpu").tolist()
                if is_symmetric:
                    k_zero = None
                    v_zero = None
                else:
                    k_zero = layer.k_zero.to("cpu").tolist()
                    v_zero = layer.v_zero.to("cpu").tolist()
            elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
                # No scales provided, use default
                k_scale = 1.0
                v_scale = 1.0
                k_zero = None  # Default to symmetric
                v_zero = None
            else:
                assert layer.k_scale > 0.0
                scale_to_duplicate = max(layer.k_scale, layer.v_scale)
                k_scale = scale_to_duplicate.to("cpu").tolist()
                v_scale = scale_to_duplicate.to("cpu").tolist()
                if is_symmetric:
                    k_zero = None
                    v_zero = None
                else:
                    zero_to_duplicate = max(layer.k_zero, layer.v_zero)
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

            layer._k_scale_rec = 1.0 / layer._k_scale
            layer._v_scale_rec = 1.0 / layer._v_scale

            # Set zero points: None for symmetric, tensor for asymmetric
            if k_zero is None:
                layer._k_zero = None
                layer._v_zero = None
                layer._k_zero_float = None
                layer._v_zero_float = None

                layer._k_zero_scaled = None
                layer._v_zero_scaled = None
            else:
                layer._k_zero = torch.tensor(
                    k_zero, dtype=layer._k_scale.dtype, device=layer._k_scale.device
                )
                layer._v_zero = torch.tensor(
                    v_zero, dtype=layer._v_scale.dtype, device=layer._v_scale.device
                )
                layer._k_zero_float = k_zero
                layer._v_zero_float = v_zero

                layer._k_zero_scaled = layer._k_zero * layer._k_scale
                layer._v_zero_scaled = layer._v_zero * layer._v_scale

        del layer.k_scale
        del layer.v_scale
        del layer.k_zero
        del layer.v_zero


class GCUInt8KVCachePerHeadMethod(GCUBaseKVCacheMethod):
    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        num_kv_heads = layer.num_kv_heads

        # Per-head scale/zero，shape: (num_kv_heads,)
        layer.k_scale = torch.nn.Parameter(
            torch.full((num_kv_heads,), -1.0, dtype=torch.float32),
            requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.full((num_kv_heads,), -1.0, dtype=torch.float32),
            requires_grad=False
        )
        layer.k_zero = torch.nn.Parameter(
            torch.zeros(num_kv_heads, dtype=torch.float32),
            requires_grad=False
        )
        layer.v_zero = torch.nn.Parameter(
            torch.zeros(num_kv_heads, dtype=torch.float32),
            requires_grad=False
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_kv_heads = layer.num_kv_heads
        device = layer._k_scale.device

        if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
            # 检查是否所有 head 的 scale 都有效（全部 > 0）
            if (layer.k_scale > 0.0).all() and (layer.v_scale > 0.0).all():
                k_scale = layer.k_scale.detach().clone()
                v_scale = layer.v_scale.detach().clone()
                k_zero = layer.k_zero.detach().clone()
                v_zero = layer.v_zero.detach().clone()
            else:
                # 有无效的 scale（< 0），使用默认值
                if (layer.k_scale > 0.0).any() or (layer.v_scale > 0.0).any():
                    logger.warning(
                        "Partial per-head KV cache scales detected (some heads "
                        "have invalid scale <= 0). Falling back to default "
                        "scale=1.0 for all heads."
                    )
                k_scale = torch.ones(num_kv_heads, dtype=torch.float32)
                v_scale = torch.ones(num_kv_heads, dtype=torch.float32)
                k_zero = torch.zeros(num_kv_heads, dtype=torch.float32)
                v_zero = torch.zeros(num_kv_heads, dtype=torch.float32)
        else:
            # kv_cache_dtype == "auto" 或 calculate_kv_scales == True
            # 使用默认值，后续可能会动态计算
            k_scale = torch.ones(num_kv_heads, dtype=torch.float32)
            v_scale = torch.ones(num_kv_heads, dtype=torch.float32)
            k_zero = torch.zeros(num_kv_heads, dtype=torch.float32)
            v_zero = torch.zeros(num_kv_heads, dtype=torch.float32)

        # 重新创建正确 shape 的内部变量（替换 Attention.__init__ 中创建的 scalar tensor）
        layer._k_scale = k_scale.to(device)
        layer._v_scale = v_scale.to(device)
        layer._k_zero = k_zero.to(device)
        layer._v_zero = v_zero.to(device)

        # 删除临时的 nn.Parameter
        del layer.k_scale
        del layer.v_scale
        del layer.k_zero
        del layer.v_zero