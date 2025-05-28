def register():

    from vllm import ModelRegistry

    from vllm_gcu.models.mixtral_quant.mixtral_quant import MixtralForCausalLM

    ModelRegistry.register_model("QuantMixtralForCausalLM", MixtralForCausalLM)