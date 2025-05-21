def register():
    import contextlib

    from vllm import ModelRegistry

    from vllm_gcu.models.deepseek_moe_quant.deepseek_moe_quant import QuantDeepseekForCausalLM

    ModelRegistry.register_model("QuantDeepseekForCausalLM", QuantDeepseekForCausalLM)