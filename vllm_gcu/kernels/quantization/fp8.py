from typing import List, Optional, Callable

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEActivationFormat
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    register_moe_scaling_factors,
    rotate_flashinfer_fp8_moe_weights,
    swap_w13_to_w31)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    expert_weight_is_col_major,
    requant_weight_ue8m0_inplace)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_moe_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize)
from vllm.utils.deep_gemm import (
    get_col_major_tma_aligned_tensor,
    is_deep_gemm_e8m0_used)
from vllm._custom_ops import scaled_fp8_quant
from vllm_gcu.kernels.batched_deep_gemm_moe import BatchedDeepGemmExpertsGCU
from vllm_gcu.kernels.fused_moe import fused_topk
from vllm.platforms import current_platform

from vllm.utils import vllm_lib
from vllm.logger import init_logger

from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.kernels.quantization.utils import (
    register_gcu_quantization_config,
    register_weight_loader_v2_supported,
)
from vllm_gcu.kernels.modular_experts import TritonExpertsPad

logger = init_logger(__name__)

@register_gcu_quantization_config("fp8")
class Fp8GCUConfig(Fp8Config):

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8GCULinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8GCUMoEMethod(self, layer)
        return super().get_quant_method(layer, prefix)

    @classmethod
    def get_name(cls) -> str:
        return "fp8_gcu"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if ("quant_method" in hf_quant_cfg
                and hf_quant_cfg["quant_method"] == "fp8"
                and user_quant in ["fp8", "fp8_gcu", None]):
            return cls.get_name()
        return None


@register_weight_loader_v2_supported
class Fp8GCULinearMethod(Fp8LinearMethod):

    def _is_per_tensor_scale(self, scale: Optional[torch.Tensor]) -> bool:
        if scale is None:
            return False
        return scale.numel() == 1

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)

        # GCU FP8 per-tensor: mm op now expects weight in [N, K] format
        if not self.block_quant:
            layer.weight = Parameter(layer.weight.t().data, requires_grad=False)

        # GCU FP8 per-tensor: swap input_scale to reciprocal
        # input_scale: stores 1/scale (for quant ops)
        # input_scale_rec: stores original scale (for matmul kernel)
        if hasattr(layer, 'input_scale') and layer.input_scale is not None:
            input_scale_rec = Parameter(
                layer.input_scale.data.clone(),
                requires_grad=False
            )
            layer.register_parameter("input_scale_rec", input_scale_rec)
            layer.input_scale = Parameter(
                (1.0 / layer.input_scale.data).to(torch.float32),
                requires_grad=False
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return apply_w8a8_block_fp8_linear(
                input=x.view(self.out_dtype)
                if x.dtype != self.out_dtype else x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale if x_scale is None else x_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            )

        # Get scales
        # input_scale: stores 1/scale (for quant ops)
        # input_scale_rec: stores original scale (for matmul kernel)
        input_scale = layer.input_scale if x_scale is None else x_scale
        input_scale_rec = getattr(layer, 'input_scale_rec', None)
        weight_scale = layer.weight_scale

        # Per-tensor quantization
        if self._is_per_tensor_scale(input_scale) and self._is_per_tensor_scale(weight_scale):

            # Quantize input if needed
            if x.dtype == current_platform.fp8_dtype():
                qinput = x.view(-1, x.shape[-1])
            else:
                # Static per-tensor quantization
                qinput, input_scale = scaled_fp8_quant(
                    x.view(-1, x.shape[-1]), input_scale)

            output_shape = [*x.shape[:-1], layer.weight.shape[0]]
            output = torch.empty(
                (qinput.shape[0], layer.weight.shape[0]),
                dtype=self.out_dtype,
                device=x.device
            )

            assert input_scale_rec is not None
            torch.ops._C.cutlass_scaled_mm(
                output, qinput, layer.weight, input_scale_rec, weight_scale, bias
            )

            return output.view(*output_shape)

        return super().apply(layer, x, bias)


class Fp8GCUMoEMethod(Fp8MoEMethod):

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer,
    ):
        if (prepare_finalize.activation_format ==
                FusedMoEActivationFormat.BatchedExperts):
            max_num_tokens_per_rank = (
                prepare_finalize.max_num_tokens_per_rank())

            return BatchedDeepGemmExpertsGCU(
                max_num_tokens_per_rank, prepare_finalize.num_dispatchers(),
                self.moe_quant_config)
        else:
            return TritonExpertsPad(self.moe_quant_config)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        config = super().get_fused_moe_quant_config(layer)
        if config is None:
            return None

        if hasattr(layer, 'w13_input_scale_rec'):
            config.w13_input_scale_rec = layer.w13_input_scale_rec
        if hasattr(layer, 'w2_input_scale_rec'):
            config.w2_input_scale_rec = layer.w2_input_scale_rec

        return config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            if current_platform.is_fp8_fnuz():
                w13_weight, w13_weight_scale_inv, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale_inv,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale_inv, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv,
                        layer.w2_input_scale)
            elif self.flashinfer_moe_backend is not None:
                # NOTE: weights have to be swapped since the activation is
                # applied on different half for flashinfer vs vllm
                w13_weight = swap_w13_to_w31(layer.w13_weight.data)
                w13_weight_scale_inv = swap_w13_to_w31(
                    layer.w13_weight_scale_inv.data)
                w2_weight = layer.w2_weight.data
                w2_weight_scale_inv = layer.w2_weight_scale_inv.data
            else:
                w13_weight = layer.w13_weight.data
                w13_weight_scale_inv = layer.w13_weight_scale_inv.data
                w2_weight = layer.w2_weight
                w2_weight_scale_inv = layer.w2_weight_scale_inv

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                                   requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                                  requires_grad=False)

            # DeepGemm scales need to be transposed and aligned. We try to do
            # it ahead of time for performance reasons.
            if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
                if expert_weight_is_col_major(layer.w13_weight_scale_inv):
                    layer.w13_weight_scale_inv = \
                        get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
                if expert_weight_is_col_major(layer.w2_weight_scale_inv):
                    layer.w2_weight_scale_inv = \
                        get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)

        # If checkpoint is fp16, quantize in place.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data,
                                          dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
                layer.local_num_experts,
                dtype=torch.float32,
                device=w13_weight.device),
                                                        requires_grad=False)
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w2_weight.data[expert, :, :])
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None
                        or layer.w2_input_scale is None):
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None.")
                if (not all_close_1d(layer.w13_input_scale)
                        or not all_close_1d(layer.w2_input_scale)):
                    logger.warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer.")
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False)
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False)
            if current_platform.is_fp8_fnuz():
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale,
                        layer.w2_input_scale)
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                      requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False)
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                     requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                           requires_grad=False)
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False)

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = scaled_fp8_quant(
                            dq_weight, 1.0 / max_w13_scales[expert_id])
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)

            if self.flashinfer_moe_backend is not None:
                # NOTE: weights have to be swapped since the activation is
                # applied on different half for flashinfer vs vllm
                assert not self.block_quant
                register_moe_scaling_factors(layer)
                w13_weight = swap_w13_to_w31(layer.w13_weight.data)
                if self.flashinfer_moe_backend == \
                    FlashinferMoeBackend.TENSORRT_LLM:
                    rotate_flashinfer_fp8_moe_weights(w13_weight, w2_weight)
                layer.w13_weight.data = w13_weight.data

        if self.use_marlin:
            prepare_moe_fp8_layer_for_marlin(layer, False)
            # Activations not quantized for marlin.
            del layer.w13_input_scale
            del layer.w2_input_scale

        if is_deep_gemm_e8m0_used() and self.block_quant:
            assert layer.weight_block_size is not None
            # Re-quantise the expert weights so their scales are UE8M0.
            block_sz = tuple(layer.weight_block_size)
            requant_weight_ue8m0_inplace(
                layer.w13_weight.data,
                layer.w13_weight_scale_inv.data,
                block_sz,
            )
            requant_weight_ue8m0_inplace(
                layer.w2_weight.data,
                layer.w2_weight_scale_inv.data,
                block_sz,
            )

            # Ensure column-major TMA alignment expected by DeepGEMM.
            if expert_weight_is_col_major(layer.w13_weight_scale_inv):
                layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(
                    layer.w13_weight_scale_inv)
            if expert_weight_is_col_major(layer.w2_weight_scale_inv):
                layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(
                    layer.w2_weight_scale_inv)

        # GCU FP8 per-tensor: swap input_scale to reciprocal
        # w13_input_scale / w2_input_scale: stores 1/scale (for quant ops)
        # w13_input_scale_rec / w2_input_scale_rec: stores original scale (for moe kernel)
        if self.quant_config.activation_scheme == "static" and not self.use_marlin:
            if hasattr(layer, 'w13_input_scale') and layer.w13_input_scale is not None:
                layer.register_parameter("w13_input_scale_rec", torch.nn.Parameter(
                    layer.w13_input_scale.data.clone(),
                    requires_grad=False
                ))
                layer.w13_input_scale = torch.nn.Parameter(
                    (1.0 / layer.w13_input_scale.data).to(torch.float32),
                    requires_grad=False
                )

            if hasattr(layer, 'w2_input_scale') and layer.w2_input_scale is not None:
                layer.register_parameter("w2_input_scale_rec", torch.nn.Parameter(
                    layer.w2_input_scale.data.clone(),
                    requires_grad=False
                ))
                layer.w2_input_scale = torch.nn.Parameter(
                    (1.0 / layer.w2_input_scale.data).to(torch.float32),
                    requires_grad=False
                )

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
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)

        zero_expert_num = getattr(layer, 'zero_expert_num', 0)
        zero_expert_type = getattr(layer, 'zero_expert_type', None)

        # use fused_topk with renormalize, no need in future versions
        if not use_grouped_topk and e_score_correction_bias is None and custom_routing_function is None:
            custom_routing_function = fused_topk

        select_result = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
        )
        topk_weights, topk_ids, zero_expert_result = select_result

        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
        )


def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_block_fp8_supported: bool = False,
    use_aiter_and_is_supported: bool = False,
) -> torch.Tensor:
    output_dtype = input.dtype
    output_shape = [*input.shape[:-1], weight.shape[0]]
    input_2d = input.view(-1, input.shape[-1])

    if input_scale is None:
        q_input, x_scale = ops.per_token_group_quant_fp8(
            input_2d,
            block_size[1],
            dtype=current_platform.fp8_dtype(),
            column_major_scales=False,
        )
    else:
        input = input.view(current_platform.fp8_dtype())
        q_input = input
        x_scale = input_scale

    output = ops.w8a8_block_fp8_matmul(
        q_input,
        weight,
        x_scale,
        weight_scale,
        block_size,
        output_dtype=output_dtype,
        bias=None,
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


# vllm_lib.impl(
#     "apply_w8a8_block_fp8_linear",
#     apply_w8a8_block_fp8_linear,
#     dispatch_key="PrivateUse1",
# )
