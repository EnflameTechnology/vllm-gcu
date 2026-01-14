import operator
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
                query = node.kwargs["query"]
                key = node.kwargs["key"]
                getitem_nodes = self.getitem_users(node)

                if (
                    is_func(query, operator.getitem)
                    and is_func(key, operator.getitem)
                    and query.args[0] == key.args[0]
                    and is_func(query.args[0], torch.ops.aten.split_with_sizes.default)
                    and all(
                        is_func(user, torch.ops.aten.slice_scatter.default)
                        for getitem_node in getitem_nodes.values()
                        for user in getitem_node.users
                    )
                ):
                    # Pattern where query and key are slices of an mm_node.
                    # While functionalized, results at [1] and [2] are scattered
                    # back into mm_node. So after de-functionalization, we can
                    # just use mm_node directly.

                    mm_node = query.args[0].args[0]
                    for user in getitem_nodes.values():
                        for user_of_getitem in user.users:
                            if is_func(
                                user_of_getitem, torch.ops.aten.slice_scatter.default
                            ):
                                user_of_getitem.replace_all_uses_with(mm_node)
                                self._remove(user_of_getitem)
                        self._remove(user)

                    self.insert_defunctionalized(graph, node)
                    self._remove(node)

                else:
                    # Directly replace the auto_functionalize(rotary_embedding)
                    # with the inplace rotary_embedding. In theory, we shouldn't
                    # do this blindly, but in practice in vLLM it's ok. The best
                    # solution is to use auto_functionalization_v2 and then use
                    # inductor's builtin defunctionalization (reinplacing) pass.
                    self.defunctionalize(graph, node, {1: 'query', 2: 'key'})

        for node in self.nodes_to_remove:
            graph.erase_node(node)

        self.nodes_to_remove = []

        return super().__call__(graph)
