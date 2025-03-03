
from typing import List, Optional, Union
from PIL import Image
from vllm.transformers_utils.config import get_config


class VLMInput:
    def __init__(self,
                 model :str,
                 tokenizer: Optional[str]):
        self.model = model
        self.hf_config = get_config(model, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.modality = 'image'

    def get_chat_template(self):
        raise NotImplementedError

    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        raise NotImplementedError

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        raise NotImplementedError

    def get_dummy_prompt(self,
                         input_len: int,
                         placeholder: str,
                         image_feature_size: int,
                         **kwargs):
        prompt = "hi" * input_len
        special_tokens_len = len(self.tokenizer(prompt).input_ids) - input_len
        if special_tokens_len > 0:
            prompt = "hi" * (input_len - special_tokens_len)
        if placeholder:
            prompt = prompt.replace(
                "hi" * image_feature_size,
                placeholder,
                1,
            )
        return prompt

    def get_dummy_vision_data(self,
                              input_vision_shape: str,
                              **kwargs):
        vision_shapes = input_vision_shape.split(";")
        if kwargs["mm_per_prompt"] != len(vision_shapes):
            raise ValueError('mm_per_prompt must be equal to input_vision_shape group')

        pil_images = []
        for vision_shape in vision_shapes:
            input_shape = vision_shape.split(",")
            image_height = int(input_shape[-2])
            image_width = int(input_shape[-1])
            pil_image = Image.new("RGB", (image_width, image_height), color=0)
            pil_images.append(pil_image)

        mm_data = {self.modality: pil_images 
                   if kwargs["mm_per_prompt"] > 1 else pil_images[0]}
        return mm_data

    def get_stop_token_ids(self):
        return None

    def get_placeholder(self):
        image_token_id = self.hf_config.image_token_index
        image_place_holder = self.tokenizer.decode(image_token_id)
        return image_place_holder

    def get_image_feature_size(self,
                               input_vision_shape: str,
                               **kwargs):
        raise NotImplementedError

    def get_modality(self):
        return self.modality

    def dataset_pre_process(self,
                            dataset_name: str,
                            question: Union[str, List],
                            **kwargs):
        return self.get_demo_prompt(question, **kwargs)

    def dataset_pred_post_process(self,
                                  dataset_name: str,
                                  pred: str):
        parsed_pred = pred.split('###')[0].strip()
        return parsed_pred
