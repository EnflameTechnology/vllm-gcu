from typing import List, Union
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput
import logging


class GOTOCR2Input(VLMInput):
    IMAGE_FEATURE_SIZE = 256
    IMAGE_TOKEN_ID = 151859
    default_OCR = "OCR"
    IMAGE_PLACEHOLDER = "<imgpad>"

    def get_chat_template(self):
        image_pad = self.IMAGE_PLACEHOLDER * self.IMAGE_FEATURE_SIZE
        template = (
            "<|im_start|>system\nYou should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n<img>"
            + image_pad
            + "</img>\n{}: <|im_end|><|im_start|>assistant"
        )

        return template

    def get_demo_prompt(self, question: str, **kwargs):
        if not question:
            logging.warning(
                f"question is empty, set default value {self.default_OCR}"
            )
            question = self.default_OCR
        template = self.get_chat_template()
        prompt = template.format(question)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        return self.IMAGE_FEATURE_SIZE

    def get_placeholder(self):
        return self.IMAGE_PLACEHOLDER * self.IMAGE_FEATURE_SIZE

    def dataset_pre_process(
        self, dataset_name: str, question: Union[str, List], **kwargs
    ):
        if dataset_name == "fox_benchmark_data":
            return self.get_demo_prompt(self.default_OCR, **kwargs)
        return self.get_demo_prompt(question, **kwargs)

    def dataset_pred_post_process(self, dataset_name: str, pred: str):
        return pred
