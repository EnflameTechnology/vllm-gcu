from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput


class YiVLInput(VLMInput):
    def get_chat_template(self):
        template = "This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n### Human: <|im_sep|>\n{}\n### Assistant:"
        return template

    def get_demo_prompt(self, question: str, **kwargs):
        template = self.get_chat_template()
        prompt = template.format(question)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        from vllm.model_executor.models.yi_vl import IMAGE_FEATURE_SIZE

        return IMAGE_FEATURE_SIZE

    def get_placeholder(self):
        from vllm.model_executor.models.yi_vl import IMAGE_TOKEN_ID

        return self.tokenizer.decode(IMAGE_TOKEN_ID)

    def get_stop_token_ids(self):
        return self.tokenizer.encode("###")
