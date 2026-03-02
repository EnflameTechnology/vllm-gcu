#!/usr/bin/env python
# coding=utf-8
import importlib as _importlib
import os as _os
import warnings as _warnings
import contextlib
from transformers import AutoConfig
import vllm_gcu.envs as gcu_envs
from vllm_gcu import reasoning

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

def _try_register_chat_template(model_type, chat_template_path):
    from pathlib import Path
    CHAT_TEMPLATES_DIR = Path(__file__).parent
    from vllm.transformers_utils.chat_templates.registry import register_chat_template_fallback_path
    register_chat_template_fallback_path(model_type, CHAT_TEMPLATES_DIR / chat_template_path)

def register_custom_models():
    from vllm import ModelRegistry
    custom_configs = [
        ("step3v", "vllm_gcu.models.step3v.step3v_config", "Step3vConfig"),
        ("hunyuan_vl", "vllm_gcu.models.hunyuan_vl.hunyuan_vl_config", "HunYuanVLConfig"),
    ]

    for name, module_name, class_name in custom_configs:
        _try_register_config(name, module_name, class_name)

    chat_templates = [
        ['deepseek_ocr', 'deepseek_ocr/template_deepseek_ocr.jinja'],
        ['deepseek_ocr2', 'deepseek_ocr2/template_deepseek_ocr.jinja'],
        ['got_ocr2', 'template_got_ocr2.jinja'],
    ]
    for model_type, chat_template in chat_templates:
        _try_register_chat_template(model_type, chat_template)

    if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
        ModelRegistry.register_model("DeepseekV3ForCausalLM", "vllm_gcu.models.deepseek_v3.deepseek_v3_with_fused_mtp:DeepseekV3ForCausalLM")
        ModelRegistry.register_model("DeepseekV32ForCausalLM", "vllm_gcu.models.deepseek_v32_with_fused_mtp:DeepseekV3ForCausalLM")

    else:
        ModelRegistry.register_model("DeepseekV3ForCausalLM", "vllm_gcu.models.deepseek_v3.deepseek_v3:DeepseekV3ForCausalLM")
        ModelRegistry.register_model("DeepseekV32ForCausalLM", "vllm_gcu.models.deepseek_v32:DeepseekV3ForCausalLM")
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_gcu.models.deepseek_mtp:DeepSeekMTP")
    ModelRegistry.register_model("GotOcr2ForConditionalGeneration", "vllm_gcu.models.got_ocr2:GotOcr2ForConditionalGeneration")
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration", "vllm_gcu.models.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")
    ModelRegistry.register_model("Qwen3VLForConditionalGeneration", "vllm_gcu.models.qwen3_vl:Qwen3VLForConditionalGeneration")
    ModelRegistry.register_model("Qwen3NextForCausalLM", "vllm_gcu.models.qwen3_next.qwen3_next:Qwen3NextForCausalLM")
    ModelRegistry.register_model("HunYuanDenseV1ForCausalLM", "vllm_gcu.models.hunyuan_v1:HunYuanDenseV1ForCausalLM")
    ModelRegistry.register_model("HunYuanMoEV1ForCausalLM", "vllm_gcu.models.hunyuan_v1:HunYuanMoEV1ForCausalLM")
    ModelRegistry.register_model("HunYuanVLForConditionalGeneration", "vllm_gcu.models.hunyuan_vl.hunyuan_vision:HunYuanVLForConditionalGeneration")
    ModelRegistry.register_model("PaddleOCRVLForConditionalGeneration", "vllm_gcu.models.paddleocr_vl:PaddleOCRVLForConditionalGeneration")

    ModelRegistry.register_model("KeyeVL1_5ForConditionalGeneration", "vllm_gcu.models.keye_vl:KeyeVL1_5ForConditionalGeneration")#gitleaks:allow
    ModelRegistry.register_model("Qwen2VLForConditionalGeneration", "vllm_gcu.models.qwen2_vl:Qwen2VLForConditionalGeneration")
    ModelRegistry.register_model("DeepseekOCRForCausalLM", "vllm_gcu.models.deepseek_ocr.deepseek_ocr:DeepseekOCRForCausalLM")
    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", "vllm_gcu.models.deepseek_ocr2.deepseek_ocr2:DeepseekOCR2ForCausalLM")
    ModelRegistry.register_model("DeepseekForCausalLM", "vllm_gcu.models.deepseek_ocr.deepseek:DeepseekForCausalLMGCU")
    ModelRegistry.register_model("MMGPTStep3vForCausalLM", "vllm_gcu.models.step3v.mm_step1o:MMGPTStep1oForCausalLM")
    ModelRegistry.register_model("Step2MiniForCausalLM", "vllm_gcu.models.step3v.step2_mini:Step2MiniForCausalLM")
    ModelRegistry.register_model("StepVLForConditionalGeneration", "vllm_gcu.models.step3_vl.step_vl:StepVLForConditionalGeneration")
    ModelRegistry.register_model("MiMoV2FlashForCausalLM", "vllm_gcu.models.mimo_v2_flash:MiMoV2FlashForCausalLM")
    ModelRegistry.register_model("GptOssForCausalLM", "vllm_gcu.models.gpt_oss:GptOssForCausalLM")
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm_gcu.models.config import DeepseekV32ForCausalLM
    MODELS_CONFIG_MAP['DeepseekV32ForCausalLM'] = DeepseekV32ForCausalLM
