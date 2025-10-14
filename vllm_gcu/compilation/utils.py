#!/usr/bin/env python
# coding=utf-8
import torch
from vllm.compilation.fusion import QUANT_OPS

from vllm_gcu.kernels.quantization.utils import kFp8DynamicTokenGroupSym, kInt8StaticTensorSym, kInt8DynamicTensorSym


QUANT_OPS[kInt8StaticTensorSym] = torch.ops._C.static_scaled_int8_quant.default
QUANT_OPS[kInt8DynamicTensorSym] = torch.ops._C.dynamic_scaled_int8_quant.default
QUANT_OPS[kFp8DynamicTokenGroupSym] = torch.ops._C.dynamic_per_token_group_fp8_quant.default
