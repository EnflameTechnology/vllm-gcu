import torch
from typing import Union
from vllm.compilation.noop_elimination import NoOpEliminationPass


class CustomNoOpEliminationPass(NoOpEliminationPass):
    def reshape_dims_equivalent(self, dim: Union[int, torch.fx.Node],
                                i_dim: Union[int, torch.SymInt]) -> bool:
        # Case 1 and 2
        if dim == i_dim or dim == -1:
            return True
        # Case 3
        return isinstance(dim, torch.fx.Node) and (dim.meta["val"]
                                                   == i_dim)._sympy_()
