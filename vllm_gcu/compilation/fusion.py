#!/usr/bin/env python
# coding=utf-8
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import fx

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from torch._inductor.pattern_matcher import PatternMatcherPass
from torch_gcu.gcu.dynamo.topsgraph.fx_passes.post_grad import inference_patterns
from vllm.compilation.fusion import (
    FP8_DTYPE,
    FusedAddRMSNormDynamicQuantPattern,
    FusedAddRMSNormStaticQuantPattern,
    RMSNormDynamicQuantPattern,
    RMSNormStaticQuantPattern,
)
from vllm.compilation.multi_output_match import MultiOutputMatch
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import CompilationConfig
from vllm.logger import init_logger


logger = init_logger(__name__)


class GCUFusionPass(VllmInductorPass):
    _instance: "Optional[FusionPass]" = None

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
        ), "FusionPass singleton instance already exists"
        super().__init__(config)

        self.matches: List[MultiOutputMatch] = []
        self.patterns: PatternMatcherPass = inference_patterns

        for epsilon in [1e-5, 1e-6]:
            RMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)
            FusedAddRMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns, self.record_match
            )
            RMSNormDynamicQuantPattern(epsilon, FP8_DTYPE, per_tensor=False).register(
                self.patterns, self.record_match
            )
            FusedAddRMSNormDynamicQuantPattern(
                epsilon, FP8_DTYPE, per_tensor=False
            ).register(self.patterns, self.record_match)

            torch._inductor.pattern_matcher._seen_patterns.clear()

    def record_match(self, match: MultiOutputMatch) -> bool:
        self.matches.append(match)
        return False

    def process_matches(self, graph: fx.Graph):
        for match in self.matches:
            match.process()

        graph.eliminate_dead_code()
        assert all(
            node not in graph.nodes
            for match in self.matches
            for node in match.match.nodes
        )

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_fusion")

        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_pattern_match")

        # Manually process multi-output matches (and run DCE)
        self.process_matches(graph)
        logger.debug("Post-processed %s matches", len(self.matches))
        self.dump_graph(graph, "after_fusion")
        self.matches.clear()
        self.end_and_log()
