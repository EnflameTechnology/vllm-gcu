from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput


class DeepSeekVLInput(VLMInput):
    def get_chat_template(self):
        template = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>\n{}\n\nAssistant:"
        return template

    def get_demo_prompt(self, question: str, **kwargs):
        template = self.get_chat_template()
        prompt = template.format(question)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        from vllm.model_executor.models.deepseek_vl import IMAGE_FEATURE_SIZE

        return IMAGE_FEATURE_SIZE

    def get_placeholder(self):
        from vllm.model_executor.models.deepseek_vl import IMAGE_TOKEN_ID

        return self.tokenizer.decode(IMAGE_TOKEN_ID)
