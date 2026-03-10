#!/usr/bin/env python
# coding=utf-8
from abc import ABC, abstractmethod
from typing import NamedTuple
from enum import Enum

import torch
from torch._ops import OpOverload
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, register_replacement
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.compilation.inductor_pass import enable_fake_mode
from vllm.compilation.vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from vllm.compilation.fusion import empty_bf16, empty_fp32
from vllm_gcu.kernels.quantization.utils import kFp8DynamicTokenGroupSym, kInt8StaticTensorSym
from vllm_gcu.compilation.utils import QUANT_OPS

logger = init_logger(__name__)

SILU_MUL_OP = torch.ops._C.silu_and_mul.default
SILU_MUL_PAD_OP = torch.ops._C.silu_and_mul_pad.default


class ActFn(Enum):
    SILU_AND_MUL = 1
    SILU_AND_MUL_PAD = 2


class FusedActQuantKey(NamedTuple):
    quant: QuantKey
    act: ActFn

    def __str__(self):
        return f"FusedActQuantKey(act {self.act} and quant {self.quant})"


FUSED_OPS: dict[FusedActQuantKey, OpOverload] = {
    FusedActQuantKey(kFp8DynamicTokenGroupSym, ActFn.SILU_AND_MUL):
    torch.ops._C.silu_mul_per_token_group_quant.default,
    FusedActQuantKey(kFp8DynamicTokenGroupSym, ActFn.SILU_AND_MUL_PAD):
    torch.ops._C.silu_mul_per_token_group_quant_with_size.default,
    FusedActQuantKey(kInt8StaticTensorSym, ActFn.SILU_AND_MUL):
    torch.ops._C.silu_mul_static_int8_quant.default,
}


class ActivationQuantPattern(ABC):

    def __init__(self, key: FusedActQuantKey):
        self.quant_key = key.quant
        self.quant_dtype = self.quant_key.dtype

        assert self.quant_key in QUANT_OPS, f"unsupported quantization scheme {self.quant_key}"
        self.QUANT_OP = QUANT_OPS[self.quant_key]

        assert key in FUSED_OPS, f"unsupported fusion scheme {key}"
        self.FUSED_OP = FUSED_OPS[key]

    def empty_quant(self, *args, **kwargs):
        kwargs = {'dtype': self.quant_dtype, 'device': "gcu", **kwargs}
        return torch.empty(*args, **kwargs)

    @abstractmethod
    def register(self, pm_pass: PatternMatcherPass):
        raise NotImplementedError


class SiluMulPerTokenGroupQuantPattern(ActivationQuantPattern):

    def __init__(self, quant_key: QuantKey):
        key = FusedActQuantKey(quant=quant_key, act=ActFn.SILU_AND_MUL)
        super().__init__(key)
        self.group_size = self.quant_key.scale.group_shape.col

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            result: torch.Tensor,
            result_silu_mul: torch.Tensor,
            input_: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                SILU_MUL_OP,
                result=result_silu_mul,
                input=input_,
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
            result_silu_mul: torch.Tensor,
            input_: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                scale=scale,
                input=input_,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2]

        inputs = [
            self.empty_quant(5, 256),  # result
            empty_bf16(5, 256),  # result_silu_mul
            empty_bf16(5, 512),  # input
            empty_fp32(5, 2),  # scale
        ]

        register_replacement(
            pattern,
            replacement,
            inputs,
            fwd_only,
            pm_pass,
        )


class SiluMulPadPerTokenGroupQuantPattern(ActivationQuantPattern):

    def __init__(self, quant_key: QuantKey):
        key = FusedActQuantKey(quant=quant_key, act=ActFn.SILU_AND_MUL_PAD)
        super().__init__(key)
        self.group_size = self.quant_key.scale.group_shape.col

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            result_silu_mul: torch.Tensor,
            input_: torch.Tensor,
            size: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                SILU_MUL_PAD_OP,
                out=result_silu_mul,
                input=input_,
                size=size,
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
            result_silu_mul: torch.Tensor,
            input_: torch.Tensor,
            size: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                scale=scale,
                input=input_,
                size=size,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2]

        inputs = [
            self.empty_quant(5, 256),  # result
            empty_bf16(5, 256),  # result_silu_mul
            empty_bf16(5, 512),  # input
            torch.full((1, ), 5, dtype=torch.int32, device="cuda"),  # size
            empty_fp32(5, 2),  # scale
        ]

        register_replacement(
            pattern,
            replacement,
            inputs,
            fwd_only,
            pm_pass,
        )


class ActivationQuantFusionPass(VllmPatternMatcherPass):

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="custom_activation_quant_fusion_pass")

        # pattern_silu_mul_fp8 = SiluMulFp8StaticQuantPattern()
        # pattern_silu_mul_fp8.register(self.patterns)
        SiluMulPerTokenGroupQuantPattern(kFp8DynamicTokenGroupSym).register(self.patterns)
        SiluMulPadPerTokenGroupQuantPattern(kFp8DynamicTokenGroupSym).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self):
        return VllmInductorPass.hash_source(
            self, ActivationQuantPattern, SiluMulPerTokenGroupQuantPattern,
            SiluMulPadPerTokenGroupQuantPattern)
