#!/usr/bin/env python
# coding=utf-8
from unittest.mock import patch
from vllm_gcu.compilation.compiler_interface import make_compiler
from vllm_gcu.compilation.fix_functionalization import GCUFixFunctionalizationPass


patch("vllm.compilation.backends.make_compiler", make_compiler).start()
patch("vllm.compilation.pass_manager.FixFunctionalizationPass", GCUFixFunctionalizationPass).start()
