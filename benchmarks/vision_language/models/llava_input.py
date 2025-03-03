import os
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput


class LlavaInput(VLMInput):
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
