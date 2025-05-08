#!/usr/bin/env python
# coding=utf-8
import importlib as _importlib
import os as _os
import warnings as _warnings


_MODEL_LIST = ["baichuan", "cusllama", "glm", "hunyuan", "moose", "deepseek_v3", "roberta", "got_ocr2", "qwen2_5_vl", "qwen3", "qwen3_moe"]


def register_custom_models():

    from vllm.utils import resolve_obj_by_qualname

    for _model in _MODEL_LIST:
        try:
            _f = resolve_obj_by_qualname(f"vllm_gcu.models.{_model}.register")
            _f()
        except ImportError as _e:
            _warnings.warn(f"Failed to import {_model}: {_e}")
        except AttributeError as _e:
            _warnings.warn(f"No defined function 'register' in {_model}: {_e}")
