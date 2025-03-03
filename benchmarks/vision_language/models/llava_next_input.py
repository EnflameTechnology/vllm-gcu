from typing import List, Optional, Union
import os
import numpy as np
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput
from vllm.assets.video import VideoAsset
from transformers import LlavaNextVideoProcessor
import cv2

class LlavaNextVideoInput(VLMInput):
    def __init__(self, model: str,
                 tokenizer):
        super().__init__(model, tokenizer)
        self.modality = 'video'

    def get_chat_template(self):
        processor = LlavaNextVideoProcessor.from_pretrained(self.model)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "{}"},
                    {"type": "video"},
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

    def get_demo_vision_data(self, input_vision_file, **kwargs):
        from vllm.assets.video import video_to_ndarrays
        video_data = video_to_ndarrays(path=input_vision_file,
                                num_frames=kwargs["num_frames"])

        return {self.modality: video_data}

    def get_placeholder(self):
        return '<video>'

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        vision_shape = input_vision_shape.split(";")
        assert len(vision_shape) == 1, "llava-next-video only support one input dummy shape"
        input_shape = vision_shape[0].split(",")
        input_height = int(input_shape[-2])
        input_width = int(input_shape[-1])
        patch_size = self.hf_config.vision_config.patch_size
        spatial_pool_stride = self.hf_config.spatial_pool_stride

        tokens_per_frame = int((input_height / patch_size / spatial_pool_stride)
                               * (input_width / patch_size / spatial_pool_stride))

        return tokens_per_frame * kwargs["num_frames"]

    def get_dummy_vision_data(self, input_vision_shape: str, **kwargs):
        input_shape = input_vision_shape.split(",")
        image_height = int(input_shape[-2])
        image_width = int(input_shape[-1])
        pil_image = Image.new("RGB", (image_width, image_height), color=0)
        np_frame = np.array(pil_image)
        mm_data_per_video = np.repeat([np_frame], kwargs["num_frames"], axis=0)
        mm_data = {self.modality: mm_data_per_video}
        return mm_data


class LlavaNextImageInput(VLMInput):
    def get_chat_template(self):
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(self.model)
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "{}"},
                {"type": "image"},
                ],
            },
        ]
        template = processor.apply_chat_template(conversation, add_generation_prompt=True)
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
        assert len(vision_shape) == 1, "llava-next-image only support one input dummy shape"
        input_shape = vision_shape[0].split(",")
        input_height = int(input_shape[-2])
        input_width = int(input_shape[-1])

        from vllm.model_executor.models.llava_next import get_llava_next_image_feature_size
        image_feature_size = get_llava_next_image_feature_size(
            self.hf_config,input_height=input_height,input_width=input_width)
        return image_feature_size

class LlavaOnevisionInputImage(VLMInput):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.modality = "image"

    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
            <|im_start|>assistant\n"
        return prompt

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        vision_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: vision_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        vision_shape = input_vision_shape.split(";")
        assert len(vision_shape) == 1, "llava-next-image only support one input dummy shape"
        input_shape = vision_shape[0].split(",")
        input_height = int(input_shape[-2])
        input_width = int(input_shape[-1])

        from vllm.model_executor.models.llava_onevision import get_llava_onevision_image_feature_size
        vision_feature_size = get_llava_onevision_image_feature_size(
            self.hf_config,input_height=input_height,input_width=input_width)
        return vision_feature_size

    def get_placeholder(self):
        place_token_id = self.hf_config.image_token_index
        place_holder = self.tokenizer.decode(place_token_id)
        return place_holder

    def dataset_pre_process(self,
                            dataset_name: str,
                            question: Union[str, List],
                            **kwargs):
        if dataset_name == "MMMU":
            return self.get_demo_prompt(question, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}.")

class LlavaOnevisionInputVideo(VLMInput):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.modality = "video"

    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
            <|im_start|>assistant\n"
        return prompt

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        from vllm.assets.video import video_to_ndarrays
        vision_data = video_to_ndarrays(path=input_vision_file,
                                    num_frames=kwargs["num_frames"])
        return {self.modality: vision_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        from vllm.model_executor.models.llava_onevision import get_llava_onevision_video_frame_feature_size
        num_token_image_newline = 1
        tokens_per_frame = get_llava_onevision_video_frame_feature_size(self.hf_config)
        vision_feature_size = kwargs["num_frames"] * tokens_per_frame + num_token_image_newline
        return vision_feature_size

    def get_placeholder(self):
        place_token_id = self.hf_config.video_token_index
        place_holder = self.tokenizer.decode(place_token_id)
        return place_holder

    def get_dummy_vision_data(self, input_vision_shape: str, **kwargs):
        image_height = 384
        image_width = 384
        pil_image = Image.new("RGB", (image_width, image_height), color=0)
        np_frame = np.array(pil_image)
        video_data = np.repeat([np_frame], kwargs["num_frames"], axis=0)
        mm_data = {self.modality: video_data}
        return mm_data

    def dataset_pre_process(self,
                            dataset_name: str,
                            question: Union[str, List],
                            **kwargs):
        if dataset_name == "Video-MME":
            cur_msgs = ""
            images = []
            for x in question:
                if x['type'] == 'text':
                    cur_msgs += x['value']
                elif x['type'] == 'image':
                    image = cv2.imread(x['value'])[np.newaxis,:,:,:]
                    images.append(image)
            images = np.concatenate(images, axis=0)
            prompt = self.get_demo_prompt(cur_msgs, **kwargs)
            vision_data = {self.modality:images}
            return prompt, vision_data
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}.")