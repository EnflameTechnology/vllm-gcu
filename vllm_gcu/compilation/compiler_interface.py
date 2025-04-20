#!/usr/bin/env python
# coding=utf-8

from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import torch.fx as fx
from vllm.compilation.compiler_interface import EagerAdaptor
from torch._inductor import config
from torch._inductor.fx_passes.pre_grad import pre_grad_passes


class CustomEagerAdaptor(EagerAdaptor):
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: List[Any],
        compiler_config: Dict[str, Any],
        runtime_shape: Optional[int] = None,
    ) -> Tuple[Optional[Callable], Optional[Any]]:
        with config.patch(compiler_config):
            pre_grad_passes(graph, example_inputs)
        return graph, None


patcher = patch("vllm.compilation.compiler_interface.EagerAdaptor", CustomEagerAdaptor)
patcher.start()
