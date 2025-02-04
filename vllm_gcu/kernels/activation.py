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


def fatrelu_and_mul_forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward_native(x)


FatreluAndMul.forward_oot = fatrelu_and_mul_forward_oot


def silu_and_mul_forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


SiluAndMul.forward_oot = silu_and_mul_forward_oot


def mul_and_silu_forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward_native(x)


MulAndSilu.forward_oot = mul_and_silu_forward_oot


def gelu_and_mul_forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_and_mul(out, x)
    return out


GeluAndMul.forward_oot = gelu_and_mul_forward_oot


def new_gelu_forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    torch.ops._C.gelu_new(out, x)
    return out


NewGELU.forward_oot = new_gelu_forward_oot


def fast_gelu_forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    torch.ops._C.gelu_fast(out, x)
    return out


FastGELU.forward_oot = fast_gelu_forward_oot


def quick_gelu_forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    torch.ops._C.gelu_quick(out, x)
    return out


QuickGELU.forward_oot = quick_gelu_forward_oot
