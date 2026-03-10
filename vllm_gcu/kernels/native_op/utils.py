import logging
import importlib
from typing import Any, Callable, Optional
import vllm 
import torch
from vllm.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)

TORCH_NATIVE_IMPL_SPECS: dict[str, dict[str, tuple[str, str]]] = {}

def register_native(namespace: str, op_name: str):
    def _decorator(fn: Callable) -> Callable:
        module_rel = fn.__module__.split(".")[-1]
        TORCH_NATIVE_IMPL_SPECS.setdefault(namespace, {})[op_name] = (module_rel, fn.__name__)
        logger.info(f"Register torch native fallback: {op_name})")
        return fn
    return _decorator

def try_import_native_op_module(op_name: str) -> None:
    """Import only requested op module; swallow failures to avoid startup crash."""
    mod = f"vllm_gcu.kernels.native_op.{op_name}"
    try:
        importlib.import_module(mod)
    except Exception as e:
        logger.warning(f"native op python module import failed: {mod} ({e})")

def try_resolve_impl(module_rel: str, attr: str) -> Optional[Callable[..., Any]]:
    impl_prefix = "vllm_gcu.kernels.native_op"
    module_name = f"{impl_prefix}.{module_rel}.{attr}"
    try:
        return resolve_obj_by_qualname(module_name)
    except Exception as e:
        logger.warning(f"native op impl unavailable: {module_name}:{attr} ({e})")
        return None

def get_op_namespace(op_name: str, supported_list: dict[str, dict[str, tuple[str, str]]]) -> str:
    for namespace, ops in supported_list.items():
        if op_name in ops:
            return namespace
    logger.warning(f"op {op_name} not found in supported_list")
    return None 