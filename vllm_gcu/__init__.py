#!/usr/bin/env python
# coding=utf-8
from typing import Optional

import tops_extension.torch
import torch
import torch_gcu
from torch_gcu import transfer_to_gcu

import vllm_gcu.distributed
from vllm_gcu.models import register_custom_models


def register_platform_plugins() -> Optional[str]:
    return "vllm_gcu.gcu.GCUPlatform"
