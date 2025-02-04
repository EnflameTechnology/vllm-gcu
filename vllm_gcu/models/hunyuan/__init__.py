def register():
    import contextlib

    from transformers import AutoConfig
    from vllm import ModelRegistry

    from vllm_gcu.models.hunyuan.hunyuan import HunyuanForCausalLM
    from vllm_gcu.models.hunyuan.hunyuan_config import HunyuanConfig

    if "HunyuanForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("HunyuanForCausalLM", HunyuanForCausalLM)

    with contextlib.suppress(ValueError):
        AutoConfig.register("hunyuan", HunyuanConfig)


# def register_hunyuan_quant():
#     from hunyuan_quant import HunyuanForCausalLM
#     from vllm import ModelRegistry
#     ModelRegistry.register_model('HunyuanForCausalLM', HunyuanForCausalLM)

#     with contextlib.suppress(ValueError):
#         AutoConfig.register("hunyuan", HunyuanConfig)
