def register():

    from vllm import ModelRegistry

    from vllm_gcu.models.qwen3.qwen3 import Qwen3ForCausalLM

    ModelRegistry.register_model("Qwen3ForCausalLM", Qwen3ForCausalLM)
