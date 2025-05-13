from typing import List, Optional, Union
import os
import math
import numpy as np
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput
from vllm.assets.video import VideoAsset
from transformers import LlavaNextVideoProcessor
from transformers import (CLIPVisionConfig, LlavaOnevisionConfig,
                          SiglipVisionConfig)
from transformers.models.llava_onevision.modeling_llava_onevision import get_anyres_image_grid_shape

import cv2


def get_llava_onevision_video_frame_feature_size(
        hf_config: LlavaOnevisionConfig) -> int:
    # Support both CLIPVisionConfig and SiglipVisionConfig
    image_size = hf_config.vision_config.image_size
    patch_size = hf_config.vision_config.patch_size
    spatial_pool_stride = hf_config.spatial_pool_stride if hasattr(
        hf_config, "spatial_pool_stride") else 2

    height = width = image_size // patch_size
    return math.ceil(height / spatial_pool_stride) * math.ceil(
        width / spatial_pool_stride)


def _get_llava_onevision_image_unppaded_feature_size(height, width, patches,
                                                     scale_height,
                                                     scale_width):
    current_height = patches * scale_height
    current_width = patches * scale_width

    original_aspect_ratio = width / height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        new_height = int(height * (current_width / width))
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = int(width * (current_height / height))
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    unpadded_features = current_height * current_width
    newline_features = current_height

    ratio = math.sqrt(current_height * current_width / (9 * patches**2))
    if ratio > 1.1:
        unpadded_features = int(current_height // ratio) * int(
            current_width // ratio)
        newline_features = int(current_height // ratio)

    return (unpadded_features, newline_features)


def get_siglip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    # Since interpolation is applied, the image size need not be divisible
    # assert image_size % patch_size == 0
    return image_size // patch_size


def get_siglip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_siglip_patch_grid_length(image_size=image_size,
                                               patch_size=patch_size)
    return grid_length * grid_length


def get_siglip_image_feature_size(hf_config: SiglipVisionConfig) -> int:
    return get_siglip_num_patches(image_size=hf_config.image_size,
                                  patch_size=hf_config.patch_size)


def get_clip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size

def get_clip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_clip_patch_grid_length(image_size=image_size,
                                             patch_size=patch_size)
    return grid_length * grid_length


def get_clip_image_feature_size(hf_config: CLIPVisionConfig) -> int:
    return get_clip_num_patches(image_size=hf_config.image_size,
                                patch_size=hf_config.patch_size) + 1


def get_llava_onevision_image_feature_size(
    hf_config: LlavaOnevisionConfig,
    *,
    input_height: int,
    input_width: int,
) -> int:
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_patches = get_clip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = get_clip_image_feature_size(vision_config)
    elif isinstance(vision_config, SiglipVisionConfig):
        num_patches = get_siglip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = get_siglip_image_feature_size(vision_config)
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    strategy = hf_config.vision_feature_select_strategy
    if strategy == "default":
        base_feature_size -= 1
    elif strategy == "full":
        pass
    else:
        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
        image_size=(input_height, input_width),
        grid_pinpoints=hf_config.image_grid_pinpoints,
        patch_size=vision_config.image_size,
    )

    (
        unpadded_feature_size,
        newline_feature_size,
    ) = _get_llava_onevision_image_unppaded_feature_size(
        input_height, input_width, num_patches, num_patch_height,
        num_patch_width)

    return unpadded_feature_size + newline_feature_size + base_feature_size


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