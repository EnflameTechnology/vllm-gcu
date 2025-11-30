from collections.abc import Iterable
import sympy
import torch
from torch import SymInt
from typing import Union
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.fx_utils import is_func
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.logger import init_logger
logger = init_logger(__name__)

class CustomNoOpEliminationPass(NoOpEliminationPass):

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        count = 0
        # Remove no-op reshapes/views:
        for node in graph.nodes:
            if is_func(node, torch.ops.aten.reshape.default):
                # Case 1: rewrite reshape chains to reshapes on the base tensor
                input = node.args[0]
                # If the input is a reshape, rebind to that node
                if is_func(input, torch.ops.aten.reshape.default):
                    # The new input is guaranteed not to be a reshape,
                    # because we process nodes in order
                    node.update_arg(0, input.args[0])
                    if len(input.users) == 0:
                        graph.erase_node(input)
                        count += 1

            # remove reshape/slice if it produces the original shape
            if is_func(node, torch.ops.aten.reshape.default) or is_func(
                node, torch.ops.aten.slice.Tensor
            ):
                input = node.args[0]
                input_shape = input.meta["val"].shape
                output_shape = node.meta["val"].shape
                if self.all_dims_equivalent(input_shape, output_shape):
                    node.replace_all_uses_with(input)
                    graph.erase_node(node)
                    count += 1
            elif is_func(node, torch.ops.aten.slice_scatter.default):
                base, view, dim_index, start, end = node.args[:5]
                base_shape = base.meta["val"].shape
                view_shape = view.meta["val"].shape

                if self.all_dims_equivalent(base_shape, view_shape):
                    node.replace_all_uses_with(view)
                    graph.erase_node(node)
                    count += 1

        logger.debug("Removed %s no-op reshapes and slices", count)

    # ---------------------- Shape comparison helpers ----------------------
    def dims_equivalent(self, dim: int | SymInt, i_dim: int | SymInt) -> bool:
        """
        This function checks if two dimensions are equivalent.
        :param dim: The dimension arg to reshape/slice
        :param i_dim: The corresponding dimension in the input tensor
        :return: Are the dimensions equivalent?

        There are two cases in which the dimensions are equivalent:
        1. The dimensions are equal (both integers)
        2. The dimensions both correspond to the same SymInt
        """
        # Case 1
        if isinstance(i_dim, int) and isinstance(dim, int):
            return dim == i_dim
        # Case 2
        if isinstance(i_dim, SymInt) and isinstance(dim, SymInt):
            return (dim == i_dim)._sympy_() == sympy.true
        return False

    def all_dims_equivalent(
        self, dims: Iterable[int | SymInt], i_dims: Iterable[int | SymInt]
    ) -> bool:
        dims_ = list(dims)
        i_dims_ = list(i_dims)
        if len(dims_) != len(i_dims_):
            # Different ranks can't be equivalent
            return False
        return all(self.dims_equivalent(s, i_s) for s, i_s in zip(dims, i_dims))
