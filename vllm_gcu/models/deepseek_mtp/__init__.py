def register():
    import contextlib

    from vllm import ModelRegistry

    from vllm_gcu.models.deepseek_mtp.deepseek_mtp import DeepSeekMTP

    ModelRegistry.register_model("DeepSeekMTPModel", DeepSeekMTP)
