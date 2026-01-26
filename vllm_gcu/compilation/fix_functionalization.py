import operator
import torch
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fx_utils import is_func
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from vllm.logger import init_logger
logger = init_logger(__name__)

class GCUFixFunctionalizationPass(FixFunctionalizationPass):
    def _cleanup_slice_scatter_users(
        self, node: torch.fx.Node, source_input: torch.fx.Node
    ) -> bool:
        """
        If the functional output is scattered back into a tensor using slice_scatter,
        and we are defunctionalizing (so the update is inplace),
        we can remove the slice_scatter and replace it with the scatter input.

        NOTE: This optimization is specifically designed for DeepSeek 3.2
        computation graphs. We use a strict whitelist of view operators
        (split_with_sizes,unsqueeze, reshape) found in those graphs.
        For other models, we prefer to miss optimization opportunities
        (leave slice_scatter) rather than risking incorrect
        removal (mis-deleting necessary ops).
        """

        # Ensure source_input and slice_scatter input share the same storage (view)
        source_ancestors = self.get_view_ancestors(source_input)

        # Iterate over a copy since we modify the graph
        for user in list(node.users):
            if is_func(user, torch.ops.aten.slice_scatter.default):
                if user.args[1] == node:
                    # Check if slice_scatter target is related to source_input
                    target = user.args[0]
                    target_ancestors = self.get_view_ancestors(target)
                    if not (source_ancestors & target_ancestors):
                        continue

                    user.replace_all_uses_with(target)
                    self._remove(user)

                    return True

        return False

    def get_view_ancestors(self, node: torch.fx.Node) -> set[torch.fx.Node]:
        """
        Trace back the view chain to find all ancestors that share storage.
        """
        ancestors = set()
        curr = node
        # Limit depth to avoid infinite loops
        for _ in range(50):
            ancestors.add(curr)
            if not isinstance(curr, torch.fx.Node):
                break

            # Handle getitem from split/chunk/unbind
            if is_func(curr, operator.getitem):
                parent = curr.args[0]
                # is_func does not support list of targets
                split_ops = [
                    # this op used in ds 3.2
                    torch.ops.aten.split_with_sizes.default,
                ]
                if parent.op == "call_function" and parent.target in split_ops:
                    curr = parent.args[0]
                    continue
                # Not a view getitem
                break

            # Handle generic view ops
            view_ops = [
                # these ops used in ds 3.2
                torch.ops.aten.unsqueeze.default,
                torch.ops.aten.reshape.default,
            ]

            if curr.op == "call_function" and curr.target in view_ops:
                if len(curr.args) > 0 and isinstance(curr.args[0], torch.fx.Node):
                    # Reshape might be a copy if input is not contiguous.
                    # Conservatively stop if input comes from ops that likely break contiguity.
                    if curr.target == torch.ops.aten.reshape.default:
                        input_node = curr.args[0]
                        is_likely_copy = False

                        # Try to use metadata if available
                        if 'val' in input_node.meta and isinstance(input_node.meta['val'], torch.Tensor):
                             # Check if input is contiguous.
                             # Note: This is a simplification. Non-contiguous inputs CAN be reshaped 
                             # without copy in some cases, but checking is_contiguous is a safe conservative check.
                             if not input_node.meta['val'].is_contiguous():
                                 is_likely_copy = True
                        else:
                             # Fallback to op checking
                             if (
                                input_node.op == "call_function"
                                and input_node.target
                                in [
                                    torch.ops.aten.permute.default,
                                    torch.ops.aten.transpose.int,
                                    torch.ops.aten.slice.Tensor,
                                ]
                            ):
                                is_likely_copy = True

                        if is_likely_copy:
                            break

                    curr = curr.args[0]
                    continue

            # Stop if not a view op
            break
        return ancestors

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

                    kwargs = node.kwargs
                    mutated_args = {1: "query", 2: "key"}

                    # designed for DeepSeek 3.2 computation graphs
                    # remove slice_scatter op if it is not necessary
                    cleanup_count = 0
                    for idx, arg_name in mutated_args.items(): #{1：getitem_40, 2:getitem_41}
                        if idx in getitem_nodes:
                            source_input = kwargs[arg_name]
                            if self._cleanup_slice_scatter_users(
                             getitem_nodes[idx], source_input):
                                cleanup_count += 1
                    logger.debug("Removed %s slice_scatter op in GCUFixFunctionalizationPass.",
                     cleanup_count)
                    # end of designed for DeepSeek 3.2 computation graphs

                    self.defunctionalize(graph, node, mutated_args)

        for node in self.nodes_to_remove:
            graph.erase_node(node)

        self.nodes_to_remove = []

        return super().__call__(graph)
