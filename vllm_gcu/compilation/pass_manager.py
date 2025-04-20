#!/usr/bin/env python
# coding=utf-8
from torch import fx as fx
from vllm.compilation.pass_manager import PostGradPassManager


class PreGradPassManager(PostGradPassManager):
    def __call__(self, graph: fx.Graph):
        for pass_ in self.passes:
            pass_(graph)
