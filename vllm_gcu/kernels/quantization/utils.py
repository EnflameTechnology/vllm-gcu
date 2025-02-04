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
