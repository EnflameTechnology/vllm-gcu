#!/usr/bin/env python
# coding=utf-8
from unittest.mock import patch


def make_compiler(compilation_config):
    if compilation_config.use_inductor:
        from vllm_gcu.compilation.compiler_interface import CustomInductorAdaptor

        return CustomInductorAdaptor()
    else:
        from vllm.compilation.compiler_interface import EagerAdaptor

        return EagerAdaptor()


patch("vllm.compilation.backends.make_compiler", make_compiler).start()
