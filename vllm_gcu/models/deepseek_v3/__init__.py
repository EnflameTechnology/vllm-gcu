def register():
    import contextlib

    from vllm import ModelRegistry

    from vllm_gcu.models.deepseek_v3.deepseek_v3 import DeepseekV3ForCausalLM

    ModelRegistry.register_model("DeepseekV3ForCausalLM", DeepseekV3ForCausalLM)
