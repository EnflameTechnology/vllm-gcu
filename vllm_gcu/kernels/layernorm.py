#!/usr/bin/env python
# coding=utf-8

from typing import Optional, Tuple, Union

import torch
from vllm.model_executor.layers.layernorm import RMSNorm


def forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if x.numel() == 0:
        if residual is not None:
            return x, x
        return x

    if self.variance_size_override is not None:
        return self.forward_native(x, residual)

    from vllm_gcu.kernels import _custom_ops as ops

    if residual is not None:
        ops.fused_add_rms_norm(
            x,
            residual,
            self.weight.data,
            self.variance_epsilon,
        )
        return x, residual
    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        self.weight.data,
        self.variance_epsilon,
    )
    return out


RMSNorm.forward_oot = forward_oot
