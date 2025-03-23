#!/usr/bin/env python
# coding=utf-8
from logging.config import dictConfig
from typing import Any, Optional
from unittest.mock import patch

import torch_gcu  # noqa: F401
import torch_gcu.transfer_to_gcu  # noqa: F401

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest

from vllm.logger import DEFAULT_LOGGING_CONFIG, VLLM_LOGGING_LEVEL


class PatchedCompletionRequest(CompletionRequest):
    extra_args: Optional[dict[str, Any]] = None

    def to_sampling_params(self, *args, **kwargs):
        params = super().to_sampling_params(*args, **kwargs)
        params.extra_args = self.extra_args

        return params


class PatchedChatCompletionRequest(ChatCompletionRequest):
    extra_args: Optional[dict[str, Any]] = None

    def to_sampling_params(self, *args, **kwargs):
        params = super().to_sampling_params(*args, **kwargs)
        params.extra_args = self.extra_args

        return params


# patch CompletionRequest & ChatCompletionRequest
patcher1 = patch(
    "vllm.entrypoints.openai.protocol.CompletionRequest", PatchedCompletionRequest
)
patcher2 = patch(
    "vllm.entrypoints.openai.protocol.ChatCompletionRequest",
    PatchedChatCompletionRequest,
)
patcher1.start()
patcher2.start()


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
