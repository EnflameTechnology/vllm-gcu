#!/usr/bin/env python
# coding=utf-8
from unittest.mock import patch
from vllm_gcu.kernels.sampler import apply_penalties, GCUTopKTopPSampler

patch("vllm.model_executor.layers.sampler.apply_penalties",
      apply_penalties).start()

patch("vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler",
        GCUTopKTopPSampler).start()
patch("vllm.v1.sample.sampler.TopKTopPSampler",
        GCUTopKTopPSampler).start()
