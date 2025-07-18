#!/usr/bin/env python
# coding=utf-8
from unittest.mock import patch
from vllm_gcu.compilation.fusion import GCUFusionPass, GCUActivationQuantFusionPass
from vllm_gcu.compilation.fix_functionalization import GCUFixFunctionalizationPass


def make_compiler(compilation_config):
    if compilation_config.use_inductor:
        from vllm_gcu.compilation.compiler_interface import CustomInductorAdaptor

        return CustomInductorAdaptor()
    else:
        from vllm.compilation.compiler_interface import EagerAdaptor

        return EagerAdaptor()


patch("vllm.compilation.backends.make_compiler", make_compiler).start()
patch("vllm.compilation.pass_manager.FusionPass", GCUFusionPass).start()
patch("vllm.compilation.pass_manager.ActivationQuantFusionPass", GCUActivationQuantFusionPass).start()
patch("vllm.compilation.pass_manager.FixFunctionalizationPass", GCUFixFunctionalizationPass).start()
