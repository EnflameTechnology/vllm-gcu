#!/usr/bin/env python
# coding=utf-8
import importlib as _importlib
import os as _os
import warnings as _warnings
import contextlib
from transformers import AutoConfig

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
    custom_configs = [
        ("cusllama", "vllm_gcu.models.cusllama.cusllama_config", "CustomerLlaMAConfig"),
    ]

    for name, module_name, class_name in custom_configs:
        _try_register_config(name, module_name, class_name)

    ModelRegistry.register_model("CustomerLlaMAForCausalLM", "vllm_gcu.models.cusllama.cusllama:CustomerLlaMAForCausalLM")
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_gcu.models.deepseek_mtp.deepseek_mtp:DeepSeekMTP")
    ModelRegistry.register_model("DeepseekV3ForCausalLM", "vllm_gcu.models.deepseek_v3.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model("XLMRobertaForSequenceClassification", "vllm_gcu.models.roberta.roberta:RobertaForSequenceClassification")
    ModelRegistry.register_model("GotOcr2ForConditionalGeneration", "vllm_gcu.models.got_ocr2.got_ocr2:GotOcr2ForConditionalGeneration")
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration", "vllm_gcu.models.qwen2_5_vl.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")
    ModelRegistry.register_model("Qwen3ForCausalLM", "vllm_gcu.models.qwen3.qwen3:Qwen3ForCausalLM")
    ModelRegistry.register_model("Qwen3MoeForCausalLM", "vllm_gcu.models.qwen3_moe.qwen3_moe:Qwen3MoeForCausalLM")
    ModelRegistry.register_model("Glm4ForCausalLM", "vllm_gcu.models.glm4.glm4:Glm4ForCausalLM")
    ModelRegistry.register_model("QuantDeepseekForCausalLM", "vllm_gcu.models.deepseek_moe_quant.deepseek_moe_quant:QuantDeepseekForCausalLM")
    ModelRegistry.register_model("QuantMixtralForCausalLM", "vllm_gcu.models.mixtral_quant.mixtral_quant:MixtralForCausalLM")
    ModelRegistry.register_model("DeepseekVLV2ForCausalLM", "vllm_gcu.models.deepseek_vl2.deepseek_vl2:DeepseekVLV2ForCausalLMGCU")
