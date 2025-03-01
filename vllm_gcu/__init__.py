#!/usr/bin/env python
# coding=utf-8
from typing import Optional

import tops_extension.torch
import torch
import torch_gcu
from torch_gcu import transfer_to_gcu
from vllm.config import ModelConfig

import vllm_gcu.distributed
from vllm_gcu.models import register_custom_models


def register_platform_plugins() -> Optional[str]:
    return "vllm_gcu.gcu.GCUPlatform"


# TODO: delete after v0.7.3
def set_use_mla(self, v):
    pass


def get_use_mla(self):
    import vllm.envs as envs
    if not self.is_deepseek_mla or envs.VLLM_MLA_DISABLE:
        return False

    return True


setattr(ModelConfig, "use_mla", property(get_use_mla, set_use_mla))
