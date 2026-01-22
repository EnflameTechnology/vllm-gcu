import torch

from .utils import (
    get_op_namespace, 
    try_resolve_impl, 
    TORCH_NATIVE_IMPL_SPECS, 
    try_import_native_op_module
)

LIBS: dict[str, torch.library.Library] = {
    "_C": torch.library.Library("_C", "IMPL"),
    "_moe_C": torch.library.Library("_moe_C", "IMPL"),
    "_C_cache_ops": torch.library.Library("_C_cache_ops", "IMPL"),
}

def register_native_overrides(additional_config: dict) -> None:
    if not additional_config:
        return

    fallback_ops = additional_config.get("fallback_ops", [])
    if not isinstance(fallback_ops, (list, tuple)):
        fallback_ops = []

    for name in fallback_ops:
        if not isinstance(name, str):
            continue
        try_import_native_op_module(name)
        namespace = get_op_namespace(name, TORCH_NATIVE_IMPL_SPECS)
        if not namespace:
            continue
        module_rel, attr = TORCH_NATIVE_IMPL_SPECS[namespace][name]
        impl = try_resolve_impl(module_rel, attr)
        if not impl:
            continue
        LIBS[f"{namespace}"].impl(name, impl, "PrivateUse1", allow_override=True)