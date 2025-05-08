def register():

    from vllm import ModelRegistry

    from vllm_gcu.models.glm4.glm4 import Glm4ForCausalLM

    ModelRegistry.register_model("Glm4ForCausalLM", Glm4ForCausalLM)
