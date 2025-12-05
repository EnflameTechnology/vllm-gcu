#!/usr/bin/env python
# coding=utf-8

import torch
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
)


@FatreluAndMul.register_oot
class GCUFatreluAndMul(FatreluAndMul):

    def forward_oot(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_native(*args, **kwargs)


@SiluAndMul.register_oot
class GCUSiluAndMul(SiluAndMul):

    def forward_oot(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_cuda(*args, **kwargs)


@MulAndSilu.register_oot
class GCUMulAndSilu(MulAndSilu):

    def forward_oot(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_native(*args, **kwargs)


@GeluAndMul.register_oot
class GCUGeluAndMul(GeluAndMul):

    def forward_oot(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_cuda(*args, **kwargs)


@NewGELU.register_oot
class GCUNewGELU(NewGELU):

    def forward_oot(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_cuda(*args, **kwargs)


@FastGELU.register_oot
class GCUFastGELU(FastGELU):

    def forward_oot(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_cuda(*args, **kwargs)


@QuickGELU.register_oot
class GCUQuickGELU(QuickGELU):

    def forward_oot(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_cuda(*args, **kwargs)
