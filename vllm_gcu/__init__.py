#!/usr/bin/env python
# coding=utf-8
from logging.config import dictConfig
from typing import Any, Optional
from unittest.mock import patch

import torch
import torch_gcu  # noqa: F401
import torch_gcu.transfer_to_gcu  # noqa: F401


from vllm.logger import DEFAULT_LOGGING_CONFIG, VLLM_LOGGING_LEVEL


def register_platform_plugins() -> Optional[str]:
    return "vllm_gcu.gcu.GCUPlatform"


VLLM_GCU_LOGGING_PREFIX = "(Module: VLLM_GCU)"
_FORMAT = (
    f"{VLLM_GCU_LOGGING_PREFIX} %(levelname)s %(asctime)s "
    "[%(filename)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG.update(
    {
        "formatters": {
            "vllm_gcu": {
                "class": "vllm.logging_utils.NewLineFormatter",
                "datefmt": _DATE_FORMAT,
                "format": _FORMAT,
            },
        },
        "handlers": {
            "vllm_gcu": {
                "class": "logging.StreamHandler",
                "formatter": "vllm_gcu",
                "level": VLLM_LOGGING_LEVEL,
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "vllm_gcu": {
                "handlers": ["vllm_gcu"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
        "version": 1,
        "disable_existing_loggers": False,
    }
)

dictConfig(DEFAULT_LOGGING_CONFIG)
