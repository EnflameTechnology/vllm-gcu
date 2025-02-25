#!/usr/bin/env python
# coding=utf-8
from fractions import Fraction
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.gptq import (
    ExllamaState,
    GPTQConfig,
    GPTQLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)

from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.kernels.quantization.utils import (
    register_gcu_quantization_config,
    register_weight_loader_v2_supported,
)


@register_gcu_quantization_config("gptq")
class GPTQGCUConfig(GPTQConfig):
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        static_groups: bool = True,
    ) -> None:
        super().__init__(weight_bits, group_size, desc_act, lm_head_quantized)
        self.static_groups = static_groups

    def __repr__(self) -> str:
        return (
            f"GPTQGCUConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}),"
            f"lm_head_quantized={self.lm_head_quantized}"
        )

    @classmethod
    def get_name(cls) -> str:
        return "gptq_gcu"

    @classmethod
    def get_min_capability(cls) -> int:
        return 30

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQGCUConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        static_groups = cls.get_from_keys_or(config, ["static_groups"], default=True)

        return cls(weight_bits, group_size, desc_act, lm_head_quantized, static_groups)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["GPTQLinearMethod"]:
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            return GPTQGCULinearMethod(self)
        return None

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if "quant_method" in hf_quant_cfg and hf_quant_cfg["quant_method"] == "gptq":
            return cls.get_name()
        return None


@register_weight_loader_v2_supported
class GPTQGCULinearMethod(GPTQLinearMethod):
    """Linear method for GPTQ with GCU impl.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQGCUConfig):
        super().__init__(quant_config)
        self.processed = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if (
            input_size != input_size_per_partition
            and self.quant_config.group_size != -1
            and self.quant_config.weight_bits == 8
        ):
            assert self.quant_config.desc_act or self.quant_config.static_groups

            scale_and_zero_size = input_size_per_partition // group_size
            scale_and_zero_input_dim = 0
            output_size_per_partition = sum(output_partition_sizes)
            weight_loader = extra_weight_attrs.get("weight_loader")

            qweight = PackedvLLMParameter(
                data=torch.empty(
                    input_size_per_partition // self.quant_config.pack_factor,
                    output_size_per_partition,
                    dtype=torch.int32,
                ),
                input_dim=0,
                output_dim=1,
                packed_dim=0,
                packed_factor=self.quant_config.pack_factor,
                weight_loader=weight_loader,
            )

            g_idx = RowvLLMParameter(
                data=torch.tensor(
                    [
                        i // self.quant_config.group_size
                        for i in range(input_size_per_partition)
                    ],
                    dtype=torch.int32,
                ),
                input_dim=0,
                weight_loader=weight_loader,
            )
            qzeros_args = {
                "data": torch.empty(
                    scale_and_zero_size,
                    output_size_per_partition // self.quant_config.pack_factor,
                    dtype=torch.int32,
                ),
                "weight_loader": weight_loader,
            }

            weight_scale_args = {
                "data": torch.empty(
                    scale_and_zero_size,
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                "weight_loader": weight_loader,
            }

            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

            layer.register_parameter("qweight", qweight)
            layer.register_parameter("g_idx", g_idx)
            layer.register_parameter("qzeros", qzeros)
            layer.register_parameter("scales", scales)

            layer.exllama_state = ExllamaState.UNINITIALIZED
        else:
            super().create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.processed:
            return

        # for torch.compile
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)

        if self.quant_config.weight_bits == 4:
            from vllm_gcu.kernels.quantization.rearrange import (
                rearrange_uint4_int32_uint8_gptq,
            )

            qweight, qzeros = rearrange_uint4_int32_uint8_gptq(
                self,
                qweight=layer.qweight.cpu(),
                qzeros=layer.qzeros.cpu(),
                scales=layer.scales.cpu(),
            )
            # TODO: support weight shuffle
            g_idx = layer.g_idx.reshape([-1, self.quant_config.group_size])[:, 0]
            if layer.qweight.nbytes == qweight.nbytes:
                layer.qweight.data = layer.qweight.data.view(torch.uint8).reshape(
                    qweight.shape
                )
                layer.qweight.data.copy_(qweight.data)
            else:
                # TODO: reduce memory fragmentation
                layer.qweight.data = qweight.to(layer.qweight.device)
                torch.gcu.empty_cache()
            layer.qzeros.data = qzeros[g_idx.cpu()].to(layer.qweight.device)
            layer.scales.data = layer.scales.data[g_idx]
        elif self.quant_config.weight_bits == 8:
            assert torch.all(
                layer.qzeros.cpu().view(torch.int8) == 127
            ), "only support sym quant"
            layer.qzeros = None
            layer.g_idx = None
            layer.qweight.data = layer.qweight.T.contiguous().view(torch.int8) - 128
            if self.quant_config.group_size == -1:
                layer.scales.data = layer.scales.data.squeeze(0)
        else:
            raise ValueError(
                "Currently, only 4/8-bit weight quantization is "
                f"supported for GPTQ on GCU, but got {self.weight_bits} bits."
            )

        self.processed = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return ops.gptq_gemm_gcu(
            x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            layer.g_idx,
            self.quant_config.weight_bits,
            bias=bias,
            group_size=self.quant_config.group_size,
        )
