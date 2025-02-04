def register():
    import contextlib

    from transformers import AutoConfig

    from vllm import ModelRegistry

    from vllm_gcu.models.cusllama.cusllama import CustomerLlaMAForCausalLM
    from vllm_gcu.models.cusllama.cusllama_config import CustomerLlaMAConfig

    if "CustomerLlaMAForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "CustomerLlaMAForCausalLM", CustomerLlaMAForCausalLM
        )

    with contextlib.suppress(ValueError):
        AutoConfig.register("cusllama", CustomerLlaMAConfig)
