#!/usr/bin/env python
# coding=utf-8
from unittest.mock import patch
from vllm_gcu.kernels.sampler import apply_penalties

patch("vllm.model_executor.layers.sampler.apply_penalties",
      apply_penalties).start()
