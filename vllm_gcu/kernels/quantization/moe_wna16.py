#!/usr/bin/env python
# coding=utf-8
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

import torch

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    is_layer_skipped_quant,
    MoeWNA16Config,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_gcu.kernels.quantization.utils import register_gcu_quantization_config


@register_gcu_quantization_config("moe_wna16")
class MoeWNA16GCUConfig(QuantizationConfig):

    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[List[str]],
        full_config: Dict[str, Any],
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.pack_factor = Fraction(32, self.weight_bits)
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config
        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert

    @classmethod
    def get_name(cls) -> str:
        return "moe_wna16_gcu"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 30

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MoeWNA16GCUConfig":
        linear_quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        if linear_quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
        elif linear_quant_method == "awq":
            has_zp = cls.get_from_keys(config, ["zero_point"])
            modules_to_not_convert = cls.get_from_keys(
                config, ["modules_to_not_convert"]
            )
        else:
            raise ValueError("moe_wna16_gcu only support gptq and awq.")

        return cls(
            linear_quant_method,
            weight_bits,
            group_size,
            has_zp,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            if self.linear_quant_method == "gptq":
                from vllm_gcu.kernels.quantization.gptq import GPTQGCUConfig

                return GPTQGCUConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            elif self.linear_quant_method == "awq":
                from vllm_gcu.kernels.quantization.awq import AWQGCUConfig

                return AWQGCUConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            else:
                raise ValueError("moe_wna16_gcu only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            return MoeWNA16GCUMethod(self)
        return None

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if (
            MoeWNA16Config.is_moe_wna16_compatible(hf_quant_cfg)
            and user_quant == "moe_wna16_gcu"
        ):
            return cls.get_name()
        return None


class MoeWNA16GCUMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: MoeWNA16GCUConfig):
        self.quant_config = quant_config
        self.processed = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.quant_config = self.quant_config

        # Currently assuming is_k_full is always True
        # (input size per partition is the same as full input size)
        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            scales_size2 = (
                intermediate_size_per_partition // self.quant_config.group_size
            )
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})

        if self.quant_config.linear_quant_method == "gptq":
            # Fused gate_up_proj (column parallel)
            w13_qweight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size // self.quant_config.pack_factor,
                    2 * intermediate_size_per_partition,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qweight", w13_qweight)
            set_weight_attrs(w13_qweight, extra_weight_attrs)
            # down_proj (row parallel)
            w2_qweight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    intermediate_size_per_partition // self.quant_config.pack_factor,
                    hidden_size,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qweight", w2_qweight)
            set_weight_attrs(w2_qweight, extra_weight_attrs)
            # up_proj scales
            w13_scales = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    scales_size13,
                    2 * intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_scales", w13_scales)
            set_weight_attrs(w13_scales, extra_weight_attrs)
            # down_proj scales
            w2_scales = torch.nn.Parameter(
                torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_scales", w2_scales)
            set_weight_attrs(w2_scales, extra_weight_attrs)

            if self.quant_config.has_zp:
                # up_proj scales
                w13_qzeros = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        scales_size13,
                        2
                        * intermediate_size_per_partition
                        // self.quant_config.pack_factor,
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w13_qzeros", w13_qzeros)
                set_weight_attrs(w13_qzeros, extra_weight_attrs)
                # down_proj scales
                w2_qzeros = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        scales_size2,
                        hidden_size // self.quant_config.pack_factor,
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w2_qzeros", w2_qzeros)
                set_weight_attrs(w2_qzeros, extra_weight_attrs)

            w13_g_idx = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_g_idx", w13_g_idx)
            set_weight_attrs(w13_g_idx, extra_weight_attrs)
            w2_g_idx = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    intermediate_size_per_partition,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_g_idx", w2_g_idx)
            set_weight_attrs(w2_g_idx, extra_weight_attrs)

        elif self.quant_config.linear_quant_method == "awq":
            w13_qweight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    2
                    * intermediate_size_per_partition
                    // self.quant_config.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qweight", w13_qweight)
            set_weight_attrs(w13_qweight, extra_weight_attrs)
            # down_proj (row parallel)
            w2_qweight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    intermediate_size_per_partition,
                    hidden_size // self.quant_config.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qweight", w2_qweight)
            set_weight_attrs(w2_qweight, extra_weight_attrs)
            # up_proj scales
            w13_scales = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    scales_size13,
                    2 * intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_scales", w13_scales)
            set_weight_attrs(w13_scales, extra_weight_attrs)
            # down_proj scales
            w2_scales = torch.nn.Parameter(
                torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_scales", w2_scales)
            set_weight_attrs(w2_scales, extra_weight_attrs)
            # up_proj scales
            w13_qzeros = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    scales_size13,
                    2
                    * intermediate_size_per_partition
                    // self.quant_config.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)
            # down_proj scales
            w2_qzeros = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    scales_size2,
                    hidden_size // self.quant_config.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

        layer.group_size = self.quant_config.group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.processed:
            return

        if self.quant_config.linear_quant_method == "gptq":
            if self.quant_config.weight_bits == 8:
                assert torch.all(
                    layer.w13_qzeros.data.view(torch.int8) == 127
                ), "only support sym quant"
                assert torch.all(
                    layer.w2_qzeros.data.view(torch.int8) == 127
                ), "only support sym quant"
                layer.w13_qweight.data = (
                    layer.w13_qweight.swapaxes(1, 2).contiguous().view(torch.int8) - 128
                )
                layer.w2_qweight.data = (
                    layer.w2_qweight.swapaxes(1, 2).contiguous().view(torch.int8) - 128
                )
                if self.quant_config.group_size == -1:
                    layer.w13_scales.data = layer.w13_scales.data.squeeze(1)
                    layer.w2_scales.data = layer.w2_scales.data.squeeze(1)
            elif self.quant_config.weight_bits == 4:
                from vllm_gcu.kernels.quantization.rearrange import (
                    rearrange_uint4_int32_uint8_gptq,
                )

                expert_num, _, _ = layer.w13_qweight.shape
                w13 = []
                w2 = []
                for i in range(expert_num):
                    qweight13 = layer.w13_qweight[i]
                    qzeros13 = layer.w13_qzeros[i] if self.quant_config.has_zp else None
                    qweight2 = layer.w2_qweight[i]
                    qzeros2 = layer.w2_qzeros[i] if self.quant_config.has_zp else None

                    g_idx13, scales13 = layer.w13_g_idx[i], layer.w13_scales[i]
                    g_idx2, scales2 = layer.w2_g_idx[i], layer.w2_scales[i]
                    qweight13, qzeros13 = rearrange_uint4_int32_uint8_gptq(
                        self,
                        qweight=qweight13.cpu(),
                        qzeros=qzeros13.cpu() if qzeros13 else None,
                        scales=scales13.cpu(),
                    )
                    qweight2, qzeros2 = rearrange_uint4_int32_uint8_gptq(
                        self,
                        qweight=qweight2.cpu(),
                        qzeros=qzeros2.cpu() if qzeros2 else None,
                        scales=scales2.cpu(),
                    )
                    # TODO: support weight shuffle
                    assert (
                        torch.all(g_idx13[1:] - g_idx13[:-1] >= 0).item()
                        and g_idx13[-1] - g_idx13[0] + 1
                        == g_idx13.shape[0] / layer.group_size
                    ), "gcu only support g_idx is continuous."
                    g_idx13 = g_idx13.reshape([-1, layer.group_size])[:, 0]
                    assert (
                        torch.all(g_idx2[1:] - g_idx2[:-1] >= 0).item()
                        and g_idx2[-1] - g_idx2[0] + 1
                        == g_idx2.shape[0] / layer.group_size
                    ), "gcu only support g_idx is continuous."
                    g_idx2 = g_idx2.reshape([-1, layer.group_size])[:, 0]
                    w13.append([qweight13, qzeros13, scales13, g_idx13])
                    w2.append([qweight2, qzeros2, scales2, g_idx2])
                if layer.w13_qweight[0].nbytes != w13[0][0].nbytes:
                    tmp_weight = w13[0][0].T
                    layer.w13_qweight.data = torch.empty(
                        expert_num,
                        *tmp_weight.shape,
                        dtype=torch.uint8,
                        device=layer.w13_qweight.data.device,
                    )
                else:
                    tmp_weight = w13[0][0].T
                    layer.w13_qweight.data = layer.w13_qweight.data.view(
                        torch.uint8
                    ).reshape((expert_num,) + tuple(tmp_weight.shape))
                if layer.w2_qweight[0].nbytes != w2[0][0].nbytes:
                    tmp_weight = w2[0][0].T
                    layer.w2_qweight.data = torch.empty(
                        expert_num,
                        *tmp_weight.shape,
                        dtype=torch.uint8,
                        device=layer.w2_qweight.data.device,
                    )
                else:
                    tmp_weight = w2[0][0].T
                    layer.w2_qweight.data = layer.w2_qweight.data.view(
                        torch.uint8
                    ).reshape((expert_num,) + tuple(tmp_weight.shape))
                if self.quant_config.has_zp:
                    assert w13[0][1] is not None
                    assert w2[0][1] is not None
                    layer.w13_qzeros.data = torch.empty(
                        expert_num,
                        *w13[0][1].shape,
                        dtype=w13[0][1].dtype,
                        device=layer.w13_qweight.data.device,
                    )
                    layer.w2_qzeros.data = torch.empty(
                        expert_num,
                        *w2[0][1].shape,
                        dtype=w13[0][1].dtype,
                        device=layer.w2_qweight.data.device,
                    )
                for i in range(expert_num):
                    layer.w13_qweight.data[i].copy_(w13[i][0].T)
                    layer.w2_qweight.data[i].copy_(w2[i][0].T)
                    layer.w13_scales.data[i].copy_(w13[i][2])
                    layer.w2_scales.data[i].copy_(w2[i][2])
                    if self.quant_config.has_zp:
                        layer.w13_qzeros.data[i].copy_(w13[i][1])
                        layer.w2_qzeros.data[i].copy_(w2[i][1])
            else:
                raise ValueError(
                    "Currently, only 8-bit and 4-bit weight quantization is "
                    f"supported for GPTQ MOE on GCU, but got {self.weight_bits} bits."
                )
        elif self.quant_config.linear_quant_method == "awq":
            assert self.quant_config.weight_bits == 4
            from vllm_gcu.kernels.quantization.rearrange import (
                rearrange_uint4_int32_uint8_awq,
            )

            expert_num, _, _ = layer.w13_qweight.shape
            w13 = []
            w2 = []

            from vllm.config import get_current_vllm_config
            model_config = get_current_vllm_config().model_config
            parallel_config = get_current_vllm_config().parallel_config

            if model_config.hf_text_config.model_type in ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp') \
                and parallel_config.enable_expert_parallel:
                weight_in_KN = True
                zeros_in_int8 = True
            else:
                weight_in_KN = False
                zeros_in_int8 = False

            for i in range(expert_num):
                qweight13, qzeros13 = layer.w13_qweight[i], layer.w13_qzeros[i]
                qweight2, qzeros2 = layer.w2_qweight[i], layer.w2_qzeros[i]
                scales13 = layer.w13_scales[i]
                scales2 = layer.w2_scales[i]
                qweight13, qzeros13 = rearrange_uint4_int32_uint8_awq(
                    self,
                    qweight=qweight13.cpu(),
                    qzeros=qzeros13.cpu(),
                    scales=scales13.cpu(),
                    zeros_in_int8=zeros_in_int8
                )
                qweight2, qzeros2 = rearrange_uint4_int32_uint8_awq(
                    self,
                    qweight=qweight2.cpu(),
                    qzeros=qzeros2.cpu(),
                    scales=scales2.cpu(),
                    zeros_in_int8=zeros_in_int8
                )
                w13.append([qweight13, qzeros13, scales13])
                w2.append([qweight2, qzeros2, scales2])
            if layer.w13_qweight[0].nbytes != w13[0][0].nbytes:
                tmp_weight = w13[0][0].T
                layer.w13_qweight.data = torch.empty(
                    expert_num,
                    *tmp_weight.shape,
                    dtype=tmp_weight.dtype,
                    device=layer.w13_qweight.data.device,
                )
            else:
                tmp_weight = w13[0][0].T
                layer.w13_qweight.data = layer.w13_qweight.data.view(
                    tmp_weight.dtype
                ).reshape((expert_num,) + tuple(tmp_weight.shape))
            if layer.w2_qweight[0].nbytes != w2[0][0].nbytes:
                tmp_weight = w2[0][0].T
                layer.w2_qweight.data = torch.empty(
                    expert_num,
                    *tmp_weight.shape,
                    dtype=tmp_weight.dtype,
                    device=layer.w2_qweight.data.device,
                )
            else:
                tmp_weight = w2[0][0].T
                layer.w2_qweight.data = layer.w2_qweight.data.view(tmp_weight.dtype).reshape(
                    (expert_num,) + tuple(tmp_weight.shape)
                )
            layer.w13_qzeros.data = torch.empty(
                expert_num,
                *w13[0][1].shape,
                dtype=w13[0][1].dtype,
                device=layer.w13_qweight.data.device,
            )
            layer.w2_qzeros.data = torch.empty(
                expert_num,
                *w2[0][1].shape,
                dtype=w13[0][1].dtype,
                device=layer.w2_qweight.data.device,
            )
            for i in range(expert_num):
                layer.w13_qweight.data[i].copy_(w13[i][0].T)
                layer.w2_qweight.data[i].copy_(w2[i][0].T)
                layer.w13_qzeros.data[i].copy_(w13[i][1])
                layer.w2_qzeros.data[i].copy_(w2[i][1])
                layer.w13_scales.data[i].copy_(w13[i][2])
                layer.w2_scales.data[i].copy_(w2[i][2])
            if weight_in_KN:
                layer.w13_qweight.data = layer.w13_qweight.data.permute(0,2,1).contiguous().permute(0,2,1)
                layer.w2_qweight.data = layer.w2_qweight.data.permute(0,2,1).contiguous().permute(0,2,1)

        self.processed = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        assert activation == "silu", "Only SiLU activation is supported."

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        weight_bits = self.quant_config.weight_bits
        has_zp = self.quant_config.has_zp

        assert has_zp, "Op impl has bug when sym"

        fused_experts(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zp=layer.w13_qzeros if has_zp else None,
            w2_zp=layer.w2_qzeros if has_zp else None,
            block_shape=[0, layer.group_size],
        )
        return x
