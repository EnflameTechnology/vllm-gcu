import sys
import types

import vllm.device_allocator  # noqa: F401


# hack to avoid ImportError
cumem = types.ModuleType("cumem")
cumem.CuMemAllocator = None
sys.modules["vllm.device_allocator.cumem"] = cumem
