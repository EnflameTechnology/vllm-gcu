import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    ScaleDesc,
    GroupShape,
    QuantKey,
    FP8_DTYPE,
    kStaticTensorScale,
    kDynamicTensorScale,
    kDynamicTokenScale,
)
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    register_quantization_config,
)


def register_gcu_quantization_config(quantization: str):
    gcu_quantization = f"{quantization}_gcu"
    if gcu_quantization not in QUANTIZATION_METHODS:
        return register_quantization_config(gcu_quantization)


def register_weight_loader_v2_supported(cls):
    from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

    WEIGHT_LOADER_V2_SUPPORTED += [cls.__name__]
    return cls


INT8_DTYPE = torch.int8

kDynamicTokenGroupScale = ScaleDesc(torch.float32, False, GroupShape(128, 128))
kFp8DynamicTokenGroupSym = QuantKey(FP8_DTYPE,
                                    kDynamicTokenGroupScale,
                                    symmetric=True)

kInt8StaticTensorSym = QuantKey(INT8_DTYPE, kStaticTensorScale, symmetric=True)
kInt8DynamicTensorSym = QuantKey(INT8_DTYPE,
                                 kDynamicTensorScale,
                                 symmetric=True)
kInt8DynamicTokenSym = QuantKey(INT8_DTYPE, kDynamicTokenScale, symmetric=True)
