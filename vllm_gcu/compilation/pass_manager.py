#!/usr/bin/env python
# coding=utf-8

from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.compilation.reshapes import RedundantReshapesPass
from vllm.config import CompilationConfig

from vllm_gcu.compilation.fusion import GCUFusionPass


class GCUPostGradPassManager(PostGradPassManager):
    def configure(self, pass_config: CompilationConfig.PassConfig):
        self.pass_config = pass_config
        if pass_config.enable_reshape:
            self.passes += [RedundantReshapesPass(pass_config)]

        if pass_config.enable_fusion:
            self.passes += [GCUFusionPass.instance(pass_config)]

        self.fix_functionalization = FixFunctionalizationPass(pass_config)
