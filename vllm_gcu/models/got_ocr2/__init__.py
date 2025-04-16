def register():
    import contextlib

    from transformers import AutoConfig
    from vllm import ModelRegistry

    from vllm_gcu.models.got_ocr2.got_ocr2 import GotOcr2ForConditionalGeneration

    if "GotOcr2ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("GotOcr2ForConditionalGeneration", GotOcr2ForConditionalGeneration)