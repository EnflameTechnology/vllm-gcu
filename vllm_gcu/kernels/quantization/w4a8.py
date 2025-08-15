# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional

import torch

from torch.nn.modules import Module
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.kernels.quantization.utils import (
    register_gcu_quantization_config,
    register_weight_loader_v2_supported,
)


@register_gcu_quantization_config("w4a8")
class W4A8Config(QuantizationConfig):
    """Config class for MOE W4A8-fp8 quantization."""

    def __init__(self, linear_quant_method: str, weight_bits: int,
                 group_size: int, has_zp: bool, lm_head_quantized: bool,
                 modules_to_not_convert: Optional[List[str]],
                 full_config: Dict[str, Any]) -> None:
        super().__init__()

        # for moe layers
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.pack_factor = 32 // self.weight_bits
        
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config

        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert

        # for fp8 layers
        self.weight_block_size = [128, 128]
        self.is_checkpoint_fp8_serialized = True
        self.activation_scheme = "dynamic"
        self.rearranged = self.get_from_keys_or(full_config, ['rearranged'], default=False)

    @classmethod
    def get_name(cls) -> str:
        return "w4a8_gcu"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W4A8Config":
        linear_quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        has_zp = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None)

        return cls(linear_quant_method, weight_bits, group_size, has_zp,
                   lm_head_quantized, modules_to_not_convert, config)


    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm_gcu.kernels.quantization.fp8 import Fp8GCULinearMethod
            return  Fp8GCULinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return MoeW4A8Method(self)
        elif isinstance(layer, Attention):
            from vllm.model_executor.layers.quantization.fp8 import Fp8KVCacheMethod
            return Fp8KVCacheMethod(self)
        return None
    
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if (
            "quant_method" in hf_quant_cfg
            and hf_quant_cfg["quant_method"] == "w4a8"
            and user_quant in ["w4a8", "w4a8_gcu", None]
        ):
            return cls.get_name()
        return None


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


@register_weight_loader_v2_supported
class MoeW4A8Method(FusedMoEMethodBase):
    """Linear method for MOE W4A8-fp8 quantization.

    Args:
        quant_config: The MOE W4A8 quantization config.
    """

    def __init__(self, quant_config: W4A8Config):
        import vllm.model_executor.layers.fused_moe
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        setattr(vllm.model_executor.layers.fused_moe, 'fused_experts', fused_experts)
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: Module) -> None:
        if not self.quant_config.rearranged:
            assert self.quant_config.weight_bits == 4
            from vllm_gcu.kernels.quantization.rearrange import (
                rearrange_uint4_int32_uint8_awq,
            )

            expert_num, _, _ = layer.w13_qweight.shape
            w13 = []
            w2 = []

            weight_in_KN = True
            zeros_in_int8 = True
            rearrange_group = 64

            for i in range(expert_num):
                qweight13, qzeros13 = layer.w13_qweight[i], None
                qweight2, qzeros2 = layer.w2_qweight[i], None
                scales13 = layer.w13_scales[i]
                scales2 = layer.w2_scales[i]
                qweight13, qzeros13 = rearrange_uint4_int32_uint8_awq(
                    self,
                    qweight=qweight13.cpu(),
                    qzeros=qzeros13,
                    scales=scales13.cpu(),
                    rearrange_group=rearrange_group,
                    use_w4a8=True,
                    zeros_in_int8=zeros_in_int8,
                )
                qweight2, qzeros2 = rearrange_uint4_int32_uint8_awq(
                    self,
                    qweight=qweight2.cpu(),
                    qzeros=qzeros2,
                    scales=scales2.cpu(),
                    rearrange_group=rearrange_group,
                    use_w4a8=True,
                    zeros_in_int8=zeros_in_int8,
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
                layer.w2_qweight.data = layer.w2_qweight.data.view(
                    tmp_weight.dtype
                ).reshape((expert_num,) + tuple(tmp_weight.shape))

            for i in range(expert_num):
                layer.w13_qweight.data[i].copy_(w13[i][0].T)
                layer.w2_qweight.data[i].copy_(w2[i][0].T)
                layer.w13_scales.data[i].copy_(w13[i][2])
                layer.w2_scales.data[i].copy_(w2[i][2])
            if weight_in_KN:
                shape13 = layer.w13_qweight.data.shape
                layer.w13_qweight.data = (
                    layer.w13_qweight.data.permute(0, 2, 1).contiguous()
                )
                layer.w13_qweight.data = layer.w13_qweight.data.view((shape13[0], shape13[1], shape13[2]))
                shape2 = layer.w2_qweight.data.shape
                layer.w2_qweight.data = (
                    layer.w2_qweight.data.permute(0, 2, 1).contiguous()
                )
                layer.w2_qweight.data = layer.w2_qweight.data.view((shape2[0], shape2[1], shape2[2]))
        else:
            shape13 = layer.w13_qweight.data.shape
            layer.w13_qweight.data = layer.w13_qweight.data.view((shape13[0], shape13[2], shape13[1]))
            shape2 = layer.w2_qweight.data.shape
            layer.w2_qweight.data = layer.w2_qweight.data.view((shape2[0], shape2[2], shape2[1]))


    def create_weights_v1(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        layer.quant_config = self.quant_config
        group_size = self.quant_config.group_size

        layer.group_size = group_size

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({
            "quant_method": strategy,
            "is_transposed": True 
        })

        assert 'weight_loader' in extra_weight_attrs
        weight_loader = extra_weight_attrs['weight_loader']
        wrapped_weight_loader = MoeW4A8Method.get_weight_loader(
            layer, weight_loader)
        extra_weight_attrs['weight_loader'] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            2 * intermediate_size_per_partition // self.quant_config.pack_factor,
            dtype=torch.int32),
                                         requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition,
            hidden_size // self.quant_config.pack_factor,
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size // group_size,
            2 * intermediate_size_per_partition,
            dtype=torch.float32),
                                        requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            intermediate_size_per_partition // group_size,
            hidden_size,
            dtype=torch.float32),
                                       requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_input_scale = torch.nn.Parameter(torch.ones(
            1,
            dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.ones(
            1,
            dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def create_weights_v2(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        layer.quant_config = self.quant_config
        group_size = self.quant_config.group_size

        layer.group_size = group_size

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({
            "quant_method": strategy,
            "is_transposed": True
        })

        assert 'weight_loader' in extra_weight_attrs
        weight_loader = extra_weight_attrs['weight_loader']
        wrapped_weight_loader = MoeW4A8Method.get_weight_loader(
            layer, weight_loader)
        extra_weight_attrs['weight_loader'] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // self.quant_config.bit8_pack_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int8),
                                         requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition // self.quant_config.bit8_pack_factor,
            hidden_size,
            dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size // group_size,
            2 * intermediate_size_per_partition,
            dtype=torch.float32),
                                        requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            intermediate_size_per_partition // group_size,
            hidden_size,
            dtype=torch.float32),
                                       requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_input_scale = torch.nn.Parameter(torch.ones(
            1,
            dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.ones(
            1,
            dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        if self.quant_config.rearranged:
            self.create_weights_v2(layer, num_experts,
                       hidden_size, intermediate_size_per_partition,
                       params_dtype, **extra_weight_attrs)
        else:
            self.create_weights_v1(layer, num_experts,
                       hidden_size, intermediate_size_per_partition,
                       params_dtype, **extra_weight_attrs)

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
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,

    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        activation += f"_{layer.layer_name}"
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)
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
            indices_type=None,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

        return fused_experts(
                hidden_states=x,
                w1=layer.w13_qweight,
                w2=layer.w2_qweight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                use_fp8_w8a8=True, # for w4a8
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_scale=layer.w13_scales,
                w2_scale=layer.w2_scales,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=None)

    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def moe_w4a8_weight_loader(param: torch.nn.Parameter,
                                    loaded_weight: torch.Tensor,
                                    weight_name: str=None, shard_id: str=None,
                                    expert_id: int=None):
            # for input_scale
            if weight_name is None:
                assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})")

                param.data.copy_(1.0/loaded_weight)
                return

            weight_loader(param, loaded_weight, weight_name, shard_id,
                              expert_id)

        return moe_w4a8_weight_loader
