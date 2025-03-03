from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput


class QwenVLInput(VLMInput):
    def get_chat_template(self):
        template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1:<img></img>\n{}<|im_end|>\n<|im_start|>assistant\n"
        return template

    def get_demo_prompt(self, question: str, **kwargs):
        template = self.get_chat_template()
        prompt = template.format(question)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        from vllm.model_executor.models.qwen import MAX_QWEN_IMG_TOKENS

        return MAX_QWEN_IMG_TOKENS

    def get_placeholder(self):
        return "<img></img>"

    def get_stop_token_ids(self):
        return [151645, 151644]
