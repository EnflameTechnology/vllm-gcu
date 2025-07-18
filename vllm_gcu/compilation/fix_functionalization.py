import torch
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fx_utils import is_func
from torch._higher_order_ops.auto_functionalize import auto_functionalized


class GCUFixFunctionalizationPass(FixFunctionalizationPass):
    def __call__(self, graph):
        self.nodes_to_remove = []

        for node in graph.nodes:
            if not is_func(node, auto_functionalized):
                continue  # Avoid deep if-elif nesting

            if node.args[0] == torch.ops._C.rotary_embedding.default:
                self.defunctionalize(graph, node, {1: 'query', 2: 'key'})

        for node in self.nodes_to_remove:
            graph.erase_node(node)

        self.nodes_to_remove = []

        return super().__call__(graph)
