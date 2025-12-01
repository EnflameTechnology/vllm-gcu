#!/usr/bin/env python
# coding=utf-8
from unittest.mock import patch
from vllm_gcu.compilation.compiler_interface import CustomInductorAdaptor


patch("vllm.compilation.compiler_interface.InductorAdaptor", CustomInductorAdaptor).start()
