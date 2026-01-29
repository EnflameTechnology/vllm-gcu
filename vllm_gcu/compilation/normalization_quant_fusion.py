#!/usr/bin/env python
# coding=utf-8
import torch
import torch._inductor.pattern_matcher as pm
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kStaticTensorScale,
    ScaleDesc,
    GroupShape,
    QuantKey,
)
from vllm.compilation.inductor_pass import enable_fake_mode
from vllm.compilation.vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from vllm.compilation.fusion import (
    RMS_OP,
    RMS_ADD_OP,
    FUSED_OPS,
    FusedRMSQuantKey,
    RMSNormQuantPattern,
    empty_bf16,
    empty_fp32,
)
from vllm_gcu.kernels.quantization.utils import (
    kFp8DynamicTokenGroupSym,
    kInt8StaticTensorSym,
    kInt8DynamicTensorSym,
    FP8_DTYPE,
    INT8_DTYPE,
)
from vllm_gcu.compilation.utils import QUANT_OPS

logger = init_logger(__name__)


FUSED_OPS[FusedRMSQuantKey(
    kFp8DynamicTokenGroupSym,
    False)] = torch.ops._C.rms_norm_per_token_group_quant_fp8.default
FUSED_OPS[FusedRMSQuantKey(
    kFp8DynamicTokenGroupSym,
    True)] = torch.ops._C.fused_add_rms_norm_per_token_group_quant_fp8.default
FUSED_OPS[FusedRMSQuantKey(
    kInt8StaticTensorSym,
    False)] = torch.ops._C.rms_norm_static_int8_quant.default
FUSED_OPS[FusedRMSQuantKey(
    kInt8StaticTensorSym,
    True)] = torch.ops._C.fused_add_rms_norm_static_int8_quant.default


class RMSNormStaticQuantPattern(RMSNormQuantPattern):

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype,
                 symmetric=True):
        fused_key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype,
                scale=kStaticTensorScale,
                symmetric=symmetric,
            ),
        )
        super().__init__(epsilon, fused_key)

    def register(self, pm_pass: PatternMatcherPass):
        # Cannot use methods, as the self argument affects tracing
        def pattern(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at1 = auto_functionalized(
                RMS_OP,
                result=result_rms,
                input=input_,
                weight=weight,
                epsilon=self.epsilon,
            )

            if self.quant_dtype == torch.int8:
                at2 = auto_functionalized(self.QUANT_OP,
                                          result=result,
                                          input=at1[1],
                                          scale=scale,
                                          azp=None)
            else:
                at2 = auto_functionalized(self.QUANT_OP,
                                          result=result,
                                          input=at1[1],
                                          scale=scale)

            # result
            return at2[1]

        def replacement(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input_,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
            )

            # result
            return at[1]

        inputs = [
            torch.empty(5, 4, device="gcu", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # result_rms
            empty_bf16(5, 4),  # input
            empty_bf16(4),  # weight
            empty_fp32([]),  # scale
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only,
                                pm_pass)


class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype,
                 symmetric=True):
        fused_key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype,
                scale=kStaticTensorScale,
                symmetric=symmetric,
            ),
        )
        super().__init__(epsilon, fused_key)

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_ADD_OP,
                input=input_,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )
            if self.quant_dtype == torch.int8:
                at1 = auto_functionalized(
                    self.QUANT_OP,
                    result=result,
                    input=at[1],
                    scale=scale,
                    azp=None,
                )
            else:
                at1 = auto_functionalized(
                    self.QUANT_OP,
                    result=result,
                    input=at[1],
                    scale=scale,
                )

            # result, residual
            return at1[1], at[2]

        def replacement(
            result: torch.Tensor,
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input_,
                residual=residual,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
            )

            # result, residual
            return at[1], at[2]

        inputs = [
            torch.empty(5, 4, device="gcu", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(4),  # weight
            empty_fp32([]),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class RMSNormDynamicQuantPattern(RMSNormQuantPattern):

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        fused_key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype,
                scale=scale,
                symmetric=symmetric,
            ),
        )
        super().__init__(epsilon, fused_key)

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at1 = auto_functionalized(
                RMS_OP,
                result=result_rms,
                input=input_,
                weight=weight,
                epsilon=self.epsilon,
            )
            at2 = auto_functionalized(self.QUANT_OP,
                                      result=result,
                                      input=at1[1],
                                      scale=scale,
                                      scale_ub=None)

            # result, scale
            return at2[1], at2[2]

        def replacement(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input_,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=None,
            )

            # result, scale
            return at[1], at[2]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # result_rms
            empty_bf16(5, 4),  # input
            empty_bf16(4),  # weight
            empty_fp32(5, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class FusedAddRMSNormDynamicQuantPattern(RMSNormQuantPattern):

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        fused_key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype,
                scale=scale,
                symmetric=symmetric,
            ),
        )
        super().__init__(epsilon, fused_key)

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_ADD_OP,
                input=input_,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )
            at1 = auto_functionalized(self.QUANT_OP,
                                      result=result,
                                      input=at[1],
                                      scale=scale,
                                      scale_ub=None)

            # result, residual, scale
            return at1[1], at[2], at1[2]

        def replacement(
            result: torch.Tensor,
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input_,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=residual,
            )

            # result, residual, scale
            return at[1], at[3], at[2]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(4),  # weight
            empty_fp32(5, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class RMSNormPerTokenGroupQuantPattern(RMSNormQuantPattern):

    def __init__(
            self,
            epsilon: float,
            quant_dtype: torch.dtype,
            group_shape: GroupShape = GroupShape(128, 128),
            symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        fused_key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype,
                scale=scale,
                symmetric=symmetric,
            ),
        )
        super().__init__(epsilon, fused_key)
        self.group_size = group_shape.col

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_OP,
                result=result_rms,
                input=input_,
                weight=weight,
                epsilon=self.epsilon,
            )
            at1 = auto_functionalized(
                self.QUANT_OP,
                out=result,
                scale=scale,
                input=at[1],
                group_size=self.group_size,
            )

            # result, scale
            return at1[1], at1[2]

        def replacement(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                scale=scale,
                input=input_,
                weight=weight,
                epsilon=self.epsilon,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2]

        inputs = [
            torch.empty(5, 512, device="cuda",
                        dtype=self.quant_dtype),  # result
            empty_bf16(5, 512),  # result_rms
            empty_bf16(5, 512),  # input
            empty_bf16(512),  # weight
            empty_fp32(5, 4),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class FusedAddRMSNormPerTokenGroupQuantPattern(RMSNormQuantPattern):

    def __init__(
            self,
            epsilon: float,
            quant_dtype: torch.dtype,
            group_shape: GroupShape = GroupShape(128, 128),
            symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        fused_key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype,
                scale=scale,
                symmetric=symmetric,
            ),
        )
        super().__init__(epsilon, fused_key)
        self.group_size = group_shape.col

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            out: torch.Tensor,
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_ADD_OP,
                input=input_,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )
            at1 = auto_functionalized(
                self.QUANT_OP,
                out=out,
                scale=scale,
                input=at[1],
                group_size=self.group_size,
            )

            # result, residual, scale
            return at1[1], at[2], at1[2]

        def replacement(
            out: torch.Tensor,
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=out,
                residual=residual,
                scale=scale,
                input=input_,
                weight=weight,
                epsilon=self.epsilon,
                group_size=self.group_size,
            )

            # result, residual, scale
            return at[1], at[2], at[3]

        inputs = [
            torch.empty(5, 512, device="cuda",
                        dtype=self.quant_dtype),  # result
            empty_bf16(5, 512),  # input
            empty_bf16(5, 512),  # residual
            empty_bf16(512),  # weight
            empty_fp32(5, 4),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class CustomRMSNormQuantFusionPass(VllmPatternMatcherPass):

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="custom_rmsnorm_quant_fusion_pass")

        for epsilon in [1e-5, 1e-6]:
            # fusion op not supported, uncomment following patterns if supported
            # RMSNormStaticQuantPattern(epsilon,
            #                           FP8_DTYPE).register(self.patterns)
            # FusedAddRMSNormStaticQuantPattern(
            #     epsilon, FP8_DTYPE).register(self.patterns)
            RMSNormDynamicQuantPattern(epsilon,
                                       FP8_DTYPE).register(self.patterns)
            FusedAddRMSNormDynamicQuantPattern(
                epsilon, FP8_DTYPE).register(self.patterns)
            RMSNormPerTokenGroupQuantPattern(
                epsilon, FP8_DTYPE).register(self.patterns)
            FusedAddRMSNormPerTokenGroupQuantPattern(
                epsilon, FP8_DTYPE).register(self.patterns)

            # RMSNormDynamicQuantPattern(epsilon,
            #                            INT8_DTYPE).register(self.patterns)
            # FusedAddRMSNormDynamicQuantPattern(
            #     epsilon, INT8_DTYPE).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self):
        return self.hash_source(
            self,
            RMSNormQuantPattern,
            RMSNormStaticQuantPattern,
            FusedAddRMSNormStaticQuantPattern,
            RMSNormDynamicQuantPattern,
            FusedAddRMSNormDynamicQuantPattern,
            RMSNormPerTokenGroupQuantPattern,
            FusedAddRMSNormPerTokenGroupQuantPattern,
        )
