from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput
from transformers.models.llava import LlavaProcessor
import transformers.models.llava


vision_patch_size = None

class LlavaInput(VLMInput):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        
        global vision_patch_size
        vision_patch_size = self.hf_config.vision_config.patch_size
        setattr(transformers.models.llava,'LlavaProcessor', ModifyLlavaProcessor)


    def get_chat_template(self):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.model)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "{}"},
                    {"type": "image"},
                ],
            },
        ]
        template = processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        return template

    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        template = self.get_chat_template()
        prompt = template.format(question)
        return prompt

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        vision_shape = input_vision_shape.split(";")
        assert len(vision_shape) == 1
        input_shape = vision_shape[0].split(",")
        input_height = int(input_shape[-2])
        input_width = int(input_shape[-1])
        patch_size = self.hf_config.vision_config.patch_size
        assert input_height % patch_size == 0 and input_width % patch_size == 0
        return (input_height // patch_size) * (input_width // patch_size)


class ModifyLlavaProcessor(LlavaProcessor):
    def __init__(self, image_processor=None, tokenizer=None, patch_size=None, vision_feature_select_strategy=None, chat_template=None, image_token="<image>", num_additional_image_tokens=0, **kwargs):
        super().__init__(image_processor, tokenizer, patch_size, vision_feature_select_strategy, chat_template, image_token, num_additional_image_tokens, **kwargs)
        if self.patch_size is None:
            self.patch_size = vision_patch_size