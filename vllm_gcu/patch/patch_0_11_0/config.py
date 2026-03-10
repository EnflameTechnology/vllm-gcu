import inspect
import ast
import os
import textwrap
from pydantic.dataclasses import dataclass
from typing import Literal, get_args
import torch
from unittest.mock import patch

from vllm.config import VllmConfig, logger
from vllm.utils import random_uuid
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.config.compilation import (CompilationConfig, CompilationLevel,
                                     CUDAGraphMode, PassConfig)
from vllm.config.cache import CacheConfig as OldCacheConfig, CacheDType
import vllm.envs as envs


class IfStatementRemover(ast.NodeTransformer):

    def __init__(self, condition_str):
        self.condition_str = condition_str
        super().__init__()

    def visit_If(self, node):
        condition_source = ast.unparse(node.test)
        if self.condition_str in condition_source:
            return None

        self.generic_visit(node)
        return node


def remove_if_from_function(func, condition):
    source = inspect.getsource(func)
    dedented_source = textwrap.dedent(source)
    tree = ast.parse(dedented_source)

    remover = IfStatementRemover(condition_str=condition)
    modified_tree = remover.visit(tree)
    modified_source = ast.unparse(modified_tree)

    exec(modified_source, globals(), locals())
    return locals()[func.__name__]


new_init = remove_if_from_function(VllmConfig.__post_init__,
                                   'self.parallel_config.enable_dbo')

patch("vllm.config.VllmConfig.__post_init__", new_init).start()

CacheDTypeExt = Literal[CacheDType, Literal["fp8_ds_mla", "int8"]]
@dataclass
class CacheConfig(OldCacheConfig):
    cache_dtype: CacheDTypeExt = "auto"
    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in get_args(CacheDTypeExt):
            if self.cache_dtype.startswith("fp8"):
                logger.info(
                    "Using fp8 data type to store kv cache. It reduces the GPU "
                    "memory footprint and boosts the performance. "
                    "Meanwhile, it may cause accuracy drop without a proper "
                    "scaling factor.")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")
patch("vllm.config.cache.CacheConfig", CacheConfig).start()
patch("vllm.config.CacheConfig", CacheConfig).start()
patch("vllm.engine.arg_utils.CacheConfig", CacheConfig).start()