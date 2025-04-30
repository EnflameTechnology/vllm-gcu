def register():

    from vllm import ModelRegistry

    from vllm_gcu.models.qwen2_5_vl.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration", Qwen2_5_VLForConditionalGeneration)