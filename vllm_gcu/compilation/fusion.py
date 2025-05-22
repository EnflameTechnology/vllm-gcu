from enum import Enum
from typing import Callable, List, NamedTuple

import torch
import torch._inductor.pattern_matcher as pm
from compressed_tensors.quantization import FP8_DTYPE
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.fusion import (
    empty_bf16,
    empty_fp32,
    FusedRMSQuantKey,
    FusionPass,
    QuantMultiOutputMatch,
    RMS_ADD_OP,
    RMS_OP,
)
from vllm.compilation.multi_output_match import MultiOutputMatch
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import CompilationConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform


logger = init_logger(__name__)


def empty_fp16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float16, device="gcu")


class Grain(Enum):
    per_tensor = 1
    per_token = 2
    per_token_group = 3
    per_channel = 4


class QuantKey(NamedTuple):
    dtype: torch.dtype
    static: bool
    grain: Grain
    symmetric: bool = True

    def __str__(self):
        return (
            f"QuantKey({'static' if self.static else 'dynamic'},"
            f"{fx.graph.dtype_abbrs[self.dtype]},"
            f"{self.grain},"
            f"{'a' if not self.symmetric else ''}symmetric)"
        )


class ActFn(Enum):
    gelu = 1
    gelu_new = 2
    gelu_fast = 3
    gelu_tanh = 4
    gelu_mul = 5
    silu = 6
    silu_mul = 7
    silu_mul_pad = 8


SILU_MUL_OP = torch.ops._C.silu_and_mul.default
SILU_MUL_PAD_OP = torch.ops._C.silu_and_mul_pad.default


kFp8StaticTensorSym = QuantKey(FP8_DTYPE, True, Grain.per_tensor, True)
kFp8DynamicTensorSym = QuantKey(FP8_DTYPE, False, Grain.per_tensor, True)
kFp8DynamicTokenSym = QuantKey(FP8_DTYPE, False, Grain.per_token, True)
kFp8DynamicTokenGroupSym = QuantKey(FP8_DTYPE, False, Grain.per_token_group, True)
kInt8StaticTensorSym = QuantKey(torch.int8, True, Grain.per_tensor, True)
kInt8StaticTensorASym = QuantKey(torch.int8, True, Grain.per_tensor, False)
kInt8DynamicTensorSym = QuantKey(torch.int8, False, Grain.per_tensor, True)


QUANT_OPS = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,
    kFp8DynamicTokenGroupSym: torch.ops._C.dynamic_per_token_group_fp8_quant.default,
    kInt8StaticTensorSym: torch.ops._C.static_scaled_int8_quant.default,
    kInt8StaticTensorASym: torch.ops._C.static_scaled_int8_asym_quant.default,
    kInt8DynamicTensorSym: torch.ops._C.dynamic_scaled_int8_quant.default,
}


class FusedActQuantKey(NamedTuple):
    quant: QuantKey
    act: ActFn

    def __str__(self):
        return f"FusedActQuantKey(act {self.act} and {self.quant})"


FUSED_OPS = {}

FUSED_RMS_OPS = {
    FusedRMSQuantKey(
        kFp8DynamicTokenGroupSym, False
    ): torch.ops._C.rms_norm_per_token_group_quant_fp8.default,
    FusedRMSQuantKey(
        kFp8DynamicTokenGroupSym, True
    ): torch.ops._C.fused_add_rms_norm_per_token_group_quant_fp8.default,
    FusedRMSQuantKey(
        kFp8StaticTensorSym, False
    ): torch.ops._C.rms_norm_static_fp8_quant.default,
    FusedRMSQuantKey(
        kFp8StaticTensorSym, True
    ): torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,
    FusedRMSQuantKey(
        kFp8DynamicTokenSym, False
    ): torch.ops._C.rms_norm_dynamic_per_token_quant.default,
    FusedRMSQuantKey(
        kFp8DynamicTokenSym, True
    ): torch.ops._C.rms_norm_dynamic_per_token_quant.default,
    FusedRMSQuantKey(kInt8StaticTensorSym, False): torch.ops._C.rms_norm_quant.default,
    FusedRMSQuantKey(
        kInt8StaticTensorSym, True
    ): torch.ops._C.fused_add_rms_norm_quant.default,
}

FUSED_ACT_OPS = {
    FusedActQuantKey(
        kFp8DynamicTokenGroupSym, ActFn.silu_mul
    ): torch.ops._C.silu_mul_per_token_group_quant.default,
    FusedActQuantKey(
        kFp8DynamicTokenGroupSym, ActFn.silu_mul_pad
    ): torch.ops._C.silu_mul_per_token_group_quant_with_size.default,
    FusedActQuantKey(kInt8StaticTensorSym, ActFn.gelu): torch.ops._C.gelu_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorSym, ActFn.gelu_new
    ): torch.ops._C.gelu_new_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorSym, ActFn.gelu_fast
    ): torch.ops._C.gelu_fast_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorSym, ActFn.gelu_tanh
    ): torch.ops._C.gelu_tanh_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorSym, ActFn.gelu_mul
    ): torch.ops._C.gelu_mul_quant.default,
    FusedActQuantKey(kInt8StaticTensorSym, ActFn.silu): torch.ops._C.silu_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorSym, ActFn.silu_mul
    ): torch.ops._C.silu_mul_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorASym, ActFn.gelu
    ): torch.ops._C.gelu_asym_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorASym, ActFn.gelu_new
    ): torch.ops._C.gelu_new_asym_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorASym, ActFn.gelu_fast
    ): torch.ops._C.gelu_fast_asym_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorASym, ActFn.gelu_tanh
    ): torch.ops._C.gelu_tanh_asym_quant.default,
    FusedActQuantKey(
        kInt8StaticTensorASym, ActFn.silu
    ): torch.ops._C.silu_asym_quant.default,
}

FUSED_OPS.update(FUSED_RMS_OPS)
FUSED_OPS.update(FUSED_ACT_OPS)


class FusedQuantPattern:
    def __init__(self, key):
        self.quant_dtype = key.quant.dtype

        assert key.quant in QUANT_OPS, f"unsupported quantization scheme {key.quant}"
        self.QUANT_OP = QUANT_OPS[key.quant]

        assert key in FUSED_OPS, f"unsupported fused rmsnorm+quant op for {key}"
        self.FUSED_OP = FUSED_OPS[key]


class RMSNormStaticQuantPattern(FusedQuantPattern):
    def __init__(self, epsilon: float, quant_dtype: torch.dtype, symmetric=True):
        fused_key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype,
                static=True,
                grain=Grain.per_tensor,
                symmetric=symmetric,
            ),
        )
        super().__init__(fused_key)
        self.epsilon = epsilon

    def register(self, pm_pass: PatternMatcherPass):
        # Cannot use methods, as the self argument affects tracing
        def pattern(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at1 = auto_functionalized(
                RMS_OP,
                result=result_rms,
                input=input,
                weight=weight,
                epsilon=self.epsilon,
            )

            if self.quant_dtype == torch.int8:
                at2 = auto_functionalized(
                    self.QUANT_OP, result=result, input=at1[1], scale=scale, azp=None
                )
            else:
                at2 = auto_functionalized(
                    self.QUANT_OP, result=result, input=at1[1], scale=scale
                )

            # result
            return at2[1]

        def replacement(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
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

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)


class FusedAddRMSNormStaticQuantPattern(FusedQuantPattern):

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, symmetric=True):
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype,
                static=True,
                grain=Grain.per_tensor,
                symmetric=symmetric,
            ),
        )
        super().__init__(key)
        self.epsilon = epsilon

    def register(
        self,
        pm_pass: PatternMatcherPass,
        record_match: Callable[[MultiOutputMatch], bool],
    ):

        def pattern(
            result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_ADD_OP,
                input=input,
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
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
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
            extra_check=lambda m: record_match(
                self.Match(m, self.QUANT_OP, self.FUSED_OP)
            ),
        )

    class Match(QuantMultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            rms_node = self.find_auto_fn(RMS_ADD_OP)
            quant_node = self.find_auto_fn(self.QUANT_OP)

            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 1

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract the result and residual.
            # The auto_fn node returns a tuple of (None, result, residual).
            #
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.fused_add_rms_norm_static_fp8_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # residual_node_new = at[2]
            with self.inserting_after_match():
                # Missing epsilon, scalars cannot be inputs to the pattern
                kwargs = self.match.kwargs.copy()

                # 0 is always None
                fused_return_mapping = {1: (quant_node, 1), 2: (rms_node, 2)}
                self.insert_fused_node(
                    fused_return_mapping, epsilon=rms_node.kwargs["epsilon"], **kwargs
                )


class RMSNormDynamicQuantPattern(FusedQuantPattern):

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, symmetric=True):
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype,
                static=False,
                grain=Grain.per_token,
                symmetric=symmetric,
            ),
        )
        super().__init__(key)
        self.epsilon = epsilon

    def register(
        self,
        pm_pass: PatternMatcherPass,
        record_match: Callable[[MultiOutputMatch], bool],
    ):

        def pattern(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at1 = auto_functionalized(
                RMS_OP,
                result=result_rms,
                input=input,
                weight=weight,
                epsilon=self.epsilon,
            )
            at2 = auto_functionalized(
                self.QUANT_OP, result=result, input=at1[1], scale=scale, scale_ub=None
            )

            # result, scale
            return at2[1], at2[2]

        def replacement(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
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
            empty_fp32(1, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=lambda m: record_match(
                self.Match(m, self.QUANT_OP, self.FUSED_OP)
            ),
        )

    class Match(QuantMultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            rms_node = self.find_auto_fn(RMS_OP)
            quant_node = self.find_auto_fn(self.QUANT_OP)

            assert len(rms_node.users) == 1
            assert len(quant_node.users) == 2

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract the result and scale.
            # The auto_fn node returns a tuple of (None, result, scale).
            #
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.rms_norm_dynamic_per_token_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # scale_node_new = at[2]
            with self.inserting_after_match():
                # Missing epsilon, scalars cannot be inputs to the pattern
                kwargs = self.match.kwargs.copy()
                del kwargs["result_rms"]  # not used in the fused op

                fused_return_mapping = {1: (quant_node, 1), 2: (quant_node, 2)}
                self.insert_fused_node(
                    fused_return_mapping,
                    epsilon=rms_node.kwargs["epsilon"],
                    scale_ub=None,  # not used but required
                    residual=None,  # not used but required
                    **kwargs,
                )


class FusedAddRMSNormDynamicQuantPattern(FusedQuantPattern):

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        symmetric=True,
    ):
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype,
                static=False,
                grain=Grain.per_token,
                symmetric=symmetric,
            ),
        )
        super().__init__(key)
        self.epsilon = epsilon

    def register(
        self,
        pm_pass: PatternMatcherPass,
        record_match: Callable[[MultiOutputMatch], bool],
    ):

        def pattern(
            result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_ADD_OP,
                input=input,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )
            at1 = auto_functionalized(
                self.QUANT_OP, result=result, input=at[1], scale=scale, scale_ub=None
            )

            # result, residual, scale
            return at1[1], at[2], at1[2]

        def replacement(
            result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
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
            empty_fp32(1, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=lambda m: record_match(
                self.Match(m, self.QUANT_OP, self.FUSED_OP)
            ),
        )

    class Match(QuantMultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            rms_node = self.find_auto_fn(RMS_ADD_OP)
            quant_node = self.find_auto_fn(self.QUANT_OP)

            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 2

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract result, scale, and residual.
            # The auto_fn node returns a tuple (None, result, scale, residual).
            #
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.rms_norm_dynamic_per_token_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # scale_node_new = at[2]
            # residual_node_new = at[3]
            with self.inserting_after_match():
                # Missing epsilon, scalars cannot be inputs to the pattern
                kwargs = self.match.kwargs.copy()

                fused_return_mapping = {
                    1: (quant_node, 1),  # result
                    2: (quant_node, 2),  # scale
                    3: (rms_node, 2),  # residual
                }
                self.insert_fused_node(
                    fused_return_mapping,
                    epsilon=rms_node.kwargs["epsilon"],
                    scale_ub=None,  # not used but required
                    **kwargs,
                )


class CSEDynamicPerTokenQuantPattern(FusedQuantPattern):
    def __init__(self, group_size: int, quant_dtype: torch.dtype, symmetric=True):
        self.quant_dtype = quant_dtype
        quant = QuantKey(
            dtype=quant_dtype,
            static=False,
            grain=Grain.per_token_group,
            symmetric=symmetric,
        )
        self.QUANT_OP = QUANT_OPS[quant]
        self.group_size = group_size

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result_1: torch.Tensor,
            result_2: torch.Tensor,
            scale_1: torch.Tensor,
            scale_2: torch.Tensor,
            input: torch.Tensor,
        ):
            at1 = auto_functionalized(
                self.QUANT_OP,
                out=result_1,
                scale=scale_1,
                input=input,
                group_size=self.group_size,
            )
            at2 = auto_functionalized(
                self.QUANT_OP,
                out=result_2,
                scale=scale_2,
                input=input,
                group_size=self.group_size,
            )

            # result, scale
            return at1[1], at1[2], at2[1], at2[2]

        def replacement(
            result_1: torch.Tensor,
            result_2: torch.Tensor,
            scale_1: torch.Tensor,
            scale_2: torch.Tensor,
            input: torch.Tensor,
        ):
            at = auto_functionalized(
                self.QUANT_OP,
                out=result_2,
                scale=scale_2,
                input=input,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2], at[1], at[2]

        inputs = [
            torch.empty(5, 512, device="cuda", dtype=self.quant_dtype),  # result1
            torch.empty(5, 512, device="cuda", dtype=self.quant_dtype),  # result2
            empty_fp32(5, 4),  # scale1
            empty_fp32(5, 4),  # scale2
            empty_bf16(5, 512),  # input
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=lambda m: len(m.nodes) == len(list(set(m.nodes))),
            # extra_check=lambda m: record_match(
            #     self.Match(m, self.QUANT_OP, self.FUSED_OP)
            # ),
        )


class SiluMulPerTokenGroupQuantPattern(FusedQuantPattern):
    def __init__(
        self,
        group_size: int,
        quant_dtype: torch.dtype,
        symmetric=True,
    ):
        key = FusedActQuantKey(
            act=ActFn.silu_mul,
            quant=QuantKey(
                dtype=quant_dtype,
                static=False,
                grain=Grain.per_token_group,
                symmetric=symmetric,
            ),
        )
        super().__init__(key)
        self.group_size = group_size

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            result_silu_mul: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                SILU_MUL_OP,
                out=result_silu_mul,
                input=input,
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
            input: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                scale=scale,
                input=input,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2]

        inputs = [
            torch.empty(5, 256, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 256),  # result_silu_mul
            empty_bf16(5, 512),  # input
            empty_fp32(5, 2),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class SiluMulPadPerTokenGroupQuantPattern(FusedQuantPattern):
    def __init__(
        self,
        group_size: int,
        quant_dtype: torch.dtype,
        symmetric=True,
    ):
        key = FusedActQuantKey(
            act=ActFn.silu_mul_pad,
            quant=QuantKey(
                dtype=quant_dtype,
                static=False,
                grain=Grain.per_token_group,
                symmetric=symmetric,
            ),
        )
        super().__init__(key)
        self.group_size = group_size

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            result_silu_mul: torch.Tensor,
            input: torch.Tensor,
            size: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                SILU_MUL_PAD_OP,
                out=result_silu_mul,
                input=input,
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
            input: torch.Tensor,
            size: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                scale=scale,
                input=input,
                size=size,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2]

        inputs = [
            torch.empty(5, 256, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 256),  # result_silu_mul
            empty_bf16(5, 512),  # input
            torch.full((1,), 5, dtype=torch.int32, device="cuda"),  # size
            empty_fp32(5, 2),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class RMSNormPerTokenGroupQuantPattern(FusedQuantPattern):

    def __init__(
        self,
        epsilon: float,
        group_size: int,
        quant_dtype: torch.dtype,
        symmetric=True,
    ):
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype,
                static=False,
                grain=Grain.per_token_group,
                symmetric=symmetric,
            ),
        )
        super().__init__(key)
        self.epsilon = epsilon
        self.group_size = group_size

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_OP,
                result=result_rms,
                input=input,
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
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                scale=scale,
                input=input,
                weight=weight,
                epsilon=self.epsilon,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2]

        inputs = [
            torch.empty(5, 512, device="cuda", dtype=self.quant_dtype),  # result
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


class FusedAddRMSNormPerTokenGroupQuantPattern(FusedQuantPattern):

    def __init__(
        self,
        epsilon: float,
        group_size: int,
        quant_dtype: torch.dtype,
        symmetric=True,
    ):
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype,
                static=False,
                grain=Grain.per_token_group,
                symmetric=symmetric,
            ),
        )
        super().__init__(key)
        self.epsilon = epsilon
        self.group_size = group_size

    def register(
        self,
        pm_pass: PatternMatcherPass,
        record_match: Callable[[MultiOutputMatch], bool],
    ):

        def pattern(
            out: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                RMS_ADD_OP,
                input=input,
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
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=out,
                residual=residual,
                scale=scale,
                input=input,
                weight=weight,
                epsilon=self.epsilon,
                group_size=self.group_size,
            )

            # result, residual, scale
            return at[1], at[2], at[3]

        inputs = [
            torch.empty(5, 512, device="cuda", dtype=self.quant_dtype),  # result
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
            extra_check=lambda m: record_match(
                self.Match(m, self.QUANT_OP, self.FUSED_OP)
            ),
        )

    class Match(QuantMultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            rms_node = self.find_auto_fn(RMS_ADD_OP)
            quant_node = self.find_auto_fn(self.QUANT_OP)

            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 2

            with self.inserting_after_match():
                # Missing epsilon, scalars cannot be inputs to the pattern
                kwargs = self.match.kwargs.copy()

                fused_return_mapping = {
                    1: (quant_node, 1),  # result
                    2: (rms_node, 2),  # residual
                    3: (quant_node, 2),  # scale
                }
                self.insert_fused_node(
                    fused_return_mapping,
                    epsilon=rms_node.kwargs["epsilon"],
                    group_size=quant_node.kwargs["group_size"],
                    **kwargs,
                )


class GCUFusionPass(FusionPass):
    @classmethod
    def instance(cls, config: CompilationConfig.PassConfig):
        if cls._instance is None:
            cls._instance = GCUFusionPass(config)
        else:
            cls._instance.config = config
        return cls._instance

    def __init__(self, config: CompilationConfig.PassConfig):
        assert (
            self.__class__._instance is None
        ), "GCUFusionPass singleton instance already exists"
        VllmInductorPass.__init__(self, config)

        self.matches: List[MultiOutputMatch] = []
        self.pre_patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="cse_quant_pss"
        )
        self.patterns: PatternMatcherPass = PatternMatcherPass(pass_name="fusion_pass")

        if current_platform.supports_fp8():
            CSEDynamicPerTokenQuantPattern(128, FP8_DTYPE).register(self.pre_patterns)

        for epsilon in [1e-5, 1e-6]:
            if current_platform.supports_fp8():
                RMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)

                # Matches for patterns below have 2 or more outputs,
                # so we need to process them manually (see process_matches)

                # Fuse rms_norm + static fp8 quant
                FusedAddRMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(
                    self.patterns, self.record_match
                )

                # Fuse rms_norm + dynamic per-token fp8 quant
                RMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns, self.record_match)

                # Fuse fused_add_rms_norm + dynamic per-token fp8 quant
                FusedAddRMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns, self.record_match)

                RMSNormPerTokenGroupQuantPattern(epsilon, 128, FP8_DTYPE).register(
                    self.patterns
                )
                FusedAddRMSNormPerTokenGroupQuantPattern(
                    epsilon, 128, FP8_DTYPE
                ).register(self.patterns, self.record_match)

                SiluMulPerTokenGroupQuantPattern(128, FP8_DTYPE).register(self.patterns)
                SiluMulPadPerTokenGroupQuantPattern(128, FP8_DTYPE).register(
                    self.patterns
                )
            else:
                RMSNormStaticQuantPattern(epsilon, torch.int8).register(self.patterns)

                # Matches for patterns below have 2 or more outputs,
                # so we need to process them manually (see process_matches)

                # Fuse rms_norm + static fp8 quant
                FusedAddRMSNormStaticQuantPattern(epsilon, torch.int8).register(
                    self.patterns, self.record_match
                )

            torch._inductor.pattern_matcher._seen_patterns.clear()

    def __call__(self, graph):
        self.begin()
        self.dump_graph(graph, "before_fusion")

        count = self.pre_patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_pattern_match")

        count = self.patterns.apply(graph)

        self.process_matches(graph)
        logger.debug("Post-processed %s matches", count)
        graph.eliminate_dead_code()
        self.dump_graph(graph, "after_fusion")

        self.matches.clear()
        self.end_and_log()
