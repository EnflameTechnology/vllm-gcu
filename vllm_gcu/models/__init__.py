#!/usr/bin/env python
# coding=utf-8
import importlib as _importlib
import os as _os
import warnings as _warnings
import contextlib
from transformers import AutoConfig
import vllm_gcu.envs as gcu_envs

def _try_register_config(name, module_name, class_name):
    try:
        mod = _importlib.import_module(module_name)
        config_cls = getattr(mod, class_name)
        with contextlib.suppress(ValueError):
            AutoConfig.register(name, config_cls)
    except ImportError as _e:
        _warnings.warn(f"Failed to import {name}: {_e}")
    except AttributeError as _e:
        _warnings.warn(f"Failed to import {name}: {_e}")

def register_custom_models():
    from vllm import ModelRegistry
    custom_configs = []

    for name, module_name, class_name in custom_configs:
        _try_register_config(name, module_name, class_name)

    if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
        ModelRegistry.register_model("DeepseekV3ForCausalLM", "vllm_gcu.models.deepseek_v3.deepseek_v3_with_fused_mtp:DeepseekV3ForCausalLM")
    else:
        ModelRegistry.register_model("DeepseekV3ForCausalLM", "vllm_gcu.models.deepseek_v3.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_gcu.models.deepseek_mtp:DeepSeekMTP")
    ModelRegistry.register_model("GotOcr2ForConditionalGeneration", "vllm_gcu.models.got_ocr2:GotOcr2ForConditionalGeneration")
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration", "vllm_gcu.models.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")
    ModelRegistry.register_model("Qwen3NextForCausalLM", "vllm_gcu.models.qwen3_next.qwen3_next:Qwen3NextForCausalLM")
    ModelRegistry.register_model("HunYuanDenseV1ForCausalLM", "vllm_gcu.models.hunyuan_v1:HunYuanDenseV1ForCausalLM")
    ModelRegistry.register_model("HunYuanMoEV1ForCausalLM", "vllm_gcu.models.hunyuan_v1:HunYuanMoEV1ForCausalLM")
