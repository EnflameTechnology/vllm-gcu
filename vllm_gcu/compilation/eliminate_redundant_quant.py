#!/usr/bin/env python
# coding=utf-8
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, register_replacement

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.compilation.inductor_pass import enable_fake_mode
from vllm.compilation.vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from vllm.compilation.fusion import empty_bf16, empty_fp32
from vllm_gcu.kernels.quantization.utils import kFp8DynamicTokenGroupSym
from vllm_gcu.compilation.utils import QUANT_OPS

logger = init_logger(__name__)


class EliminateDynamicPerTokenQuantPattern:

    def __init__(self, quant_key: QuantKey):
        assert quant_key in QUANT_OPS
        self.QUANT_OP = QUANT_OPS[quant_key]
        self.quant_dtype = quant_key.dtype
        self.group_size = quant_key.scale.group_shape.col

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ):

        def pattern(
            result_1: torch.Tensor,
            result_2: torch.Tensor,
            scale_1: torch.Tensor,
            scale_2: torch.Tensor,
            input_: torch.Tensor,
        ):
            at1 = auto_functionalized(
                self.QUANT_OP,
                out=result_1,
                scale=scale_1,
                input=input_,
                group_size=self.group_size,
            )
            at2 = auto_functionalized(
                self.QUANT_OP,
                out=result_2,
                scale=scale_2,
                input=input_,
                group_size=self.group_size,
            )

            # result, scale
            return at1[1], at1[2], at2[1], at2[2]

        def replacement(
            result_1: torch.Tensor,
            result_2: torch.Tensor,
            scale_1: torch.Tensor,
            scale_2: torch.Tensor,
            input_: torch.Tensor,
        ):
            at = auto_functionalized(
                self.QUANT_OP,
                out=result_2,
                scale=scale_2,
                input=input_,
                group_size=self.group_size,
            )

            # result, scale
            return at[1], at[2], at[1], at[2]

        inputs = [
            torch.empty(5, 512, device="cuda",
                        dtype=self.quant_dtype),  # result1
            torch.empty(5, 512, device="cuda",
                        dtype=self.quant_dtype),  # result2
            empty_fp32(5, 4),  # scale1
            empty_fp32(5, 4),  # scale2
            empty_bf16(5, 512),  # input
        ]

        register_replacement(
            pattern,
            replacement,
            inputs,
            fwd_only,
            pm_pass,
            extra_check=lambda m: len(m.nodes) == len(list(set(m.nodes))),
        )


class EliminateRedundantQuantPass(VllmPatternMatcherPass):

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="eliminate_redundant_quant_pass")

        EliminateDynamicPerTokenQuantPattern(kFp8DynamicTokenGroupSym).register(
            self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self):
        return VllmInductorPass.hash_source(
            self, EliminateDynamicPerTokenQuantPattern)
