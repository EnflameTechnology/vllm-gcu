from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput


class GLM4VInput(VLMInput):
    IMAGE_FEATURE_SIZE = 1602
    IMAGE_TOKEN_ID = 151339

    def get_chat_template(self):
        template = (
            "[gMASK] <sop> <|user|>\n <|begin_of_image|>\n{}\n<|assistant|>"
        )
        return template

    def get_demo_prompt(self, question: str, **kwargs):
        template = self.get_chat_template()
        prompt = template.format(question)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        return self.IMAGE_FEATURE_SIZE

    def get_placeholder(self):
        return self.tokenizer.decode(self.IMAGE_TOKEN_ID)
