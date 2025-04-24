import importlib

VLM_MODELS = {
    "LlavaNextVideoForConditionalGeneration": ("llava_next_input", "LlavaNextVideoInput"),
    "LlavaNextForConditionalGeneration": ("llava_next_input", "LlavaNextImageInput"),
    "LlavaOnevisionForConditionalGenerationImage": ("llava_next_input", "LlavaOnevisionInputImage"),
    "LlavaOnevisionForConditionalGenerationVideo": ("llava_next_input", "LlavaOnevisionInputVideo"),
    "LlavaForConditionalGeneration": ("llava_input", "LlavaInput"),
    "InternVLChatModel": ("internvl_input", "InternVL2Input"),
    "InternVLChatModelVideo": ("internvl_input", "InternVL2VideoInput"),
    "LlavaLlamaForCausalLM": ("yi_vl_input", "YiVLInput"),
    "QWenLMHeadModel": ("qwen_vl_input","QwenVLInput"),
    "ChatGLMModel": ("glm4v_input", "GLM4VInput"),
    "MultiModalityCausalLM": ("deepseek_vl_input","DeepSeekVLInput"),
    "Phi3VForCausalLMImage": ("phi3v_input", "Phi3VImageInput"),
    "Phi3VForCausalLMVideo": ("phi3v_input", "Phi3VVideoInput"),
    "Qwen2VLForConditionalGenerationImage": ("qwen2_vl_input", "Qwen2VLImageInput"),
    "Qwen2VLForConditionalGenerationVideo": ("qwen2_vl_input", "Qwen2VLVideoInput"),
    "Qwen2_5_VLForConditionalGenerationImage": ("qwen2_vl_input", "Qwen2VLImageInput"),
    "Qwen2_5_VLForConditionalGenerationVideo": ("qwen2_vl_input", "Qwen2VLVideoInput"),
    "MiniCPMVImage": ("minicpmv_input", "MinicpmvImageInput"),
    "MiniCPMVVideo": ("minicpmv_input", "MinicpmvVideoInput"),
    "DeepseekVLV2ForCausalLM": ("deepseek_vl2_input", "DeepSeekVL2Input"),
    "GotOcr2ForConditionalGeneration": ("got_ocr", "GOTOCR2Input"),

}


def resolve_vlm_input_cls(model_arch: str):
    module_name, model_cls_name = VLM_MODELS[model_arch]
    module = importlib.import_module(
        f"vllm_utils.vision_language.models.{module_name}")
    return getattr(module, model_cls_name, None)
