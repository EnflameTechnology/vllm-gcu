def register():

    from vllm import ModelRegistry

    from vllm_gcu.models.qwen3_moe.qwen3_moe import Qwen3MoeForCausalLM

    ModelRegistry.register_model("Qwen3MoeForCausalLM", Qwen3MoeForCausalLM)
