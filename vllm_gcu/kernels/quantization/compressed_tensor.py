
from typing import Optional, Tuple

import torch

from vllm.platforms import current_platform

from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import CutlassScaledMMLinearKernel
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel, ScaledMMLinearLayerConfig)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import _POSSIBLE_KERNELS
from vllm.platforms import PlatformEnum

class CompressedTensorMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 30

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if not c.input_symmetric:
            return (False,
                    "CompressedTensorMMLinearKernel only supports symmetric " +
                    "quantization.")
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().apply_weights(layer, x, bias)


_POSSIBLE_KERNELS.update({
    PlatformEnum.OOT: [CompressedTensorMMLinearKernel],
})