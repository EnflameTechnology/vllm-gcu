#!/usr/bin/env python
# coding=utf-8
import torch
from vllm.compilation.inductor_pass import InductorPass, get_pass_context
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.compilation.pass_manager import with_pattern_match_debug, PostGradPassManager
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm_gcu.compilation.eliminate_redundant_quant import EliminateRedundantQuantPass
from vllm_gcu.compilation.activation_quant_fusion import ActivationQuantFusionPass
from vllm_gcu.compilation.normalization_quant_fusion import CustomRMSNormQuantFusionPass
from vllm_gcu.compilation.noop_elimination import CustomNoOpEliminationPass
from vllm_gcu.compilation.replace_full_with_empty import replace_full_with_empty


class SingletonDescriptor:

    def __init__(self, cls):
        self.cls = cls
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self.cls(*args, **kwargs)
        return self._instance

    def __getattr__(self, name):
        return getattr(self.cls, name)


@SingletonDescriptor
class SingletonPostGradPassManager(PostGradPassManager):
    pass


def fallback_prims(graph):
    to_remove = []
    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.prims.convert_element_type.default:
            to_remove.append(node)

            input_tensor = node.args[0]
            target_dtype = node.args[1]

            with graph.inserting_after(node):
                new_node = graph.call_function(torch.ops.aten.to.dtype,
                                               (input_tensor, target_dtype),
                                               {})

            node.replace_all_uses_with(new_node)

    for node in to_remove:
        graph.erase_node(node)


class PassManager(InductorPass):

    def __init__(self, config):
        self.is_deepseek_mla = config.model_config.is_deepseek_mla

        self.passes: list[VllmInductorPass] = []
        self.passes += [CustomNoOpEliminationPass(config)]
        self.passes += [EliminateRedundantQuantPass(config)]
        self.passes += [CustomRMSNormQuantFusionPass(config)]
        self.passes += [ActivationQuantFusionPass(config)]

    @with_pattern_match_debug
    def __call__(self, graph: torch.fx.Graph):
        VllmInductorPass.dump_prefix = 0

        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)
                VllmInductorPass.dump_prefix += 1

        fallback_prims(graph)
        if self.is_deepseek_mla:
            replace_full_with_empty(graph)

        from torch._inductor.pattern_matcher import stable_topological_sort
        stable_topological_sort(graph)
        graph.eliminate_dead_code()

    def __str__(self):
        return 'PassManager()'
