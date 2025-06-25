import torch
from typing import Union
from torch import SymInt
from unittest.mock import patch


def dims_equivalent(self, dim: Union[int, torch.fx.Node],
                    i_dim: Union[int, SymInt]) -> bool:
    # Case 1 and 2
    if dim == i_dim or dim == -1:
        return True
    # Case 3
    return isinstance(dim, torch.fx.Node) and (dim.meta["val"] == i_dim)._sympy_() == True


patch('vllm.compilation.noop_elimination.NoOpEliminationPass.dims_equivalent', dims_equivalent).start()
