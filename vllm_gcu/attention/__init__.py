from vllm.utils import vllm_lib


for opdef, handle in zip(vllm_lib._op_defs, vllm_lib._registration_handles):
    if opdef == "vllm::unified_attention":
        handle.destroy()
