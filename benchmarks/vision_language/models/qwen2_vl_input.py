import os
import logging
import numpy as np
import cv2

from PIL import Image
from typing import Optional, List, Union
from transformers import AutoProcessor

from vllm.assets.video import video_to_ndarrays
from vllm_utils.vision_language.models.base import VLMInput


def _get_vision_info(
    image_processor,
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    do_resize: bool = True,
    data_type_key: str = "image",
    mm_count: int = 1,
):
    """Get information (resized height / width and number of vision tokens)
    of input image / video frame."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    if do_resize:
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            factor=image_processor.patch_size * image_processor.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        resized_height, resized_width = height, width

    if data_type_key == "image":
        grid_t = mm_count
    else:
        assert data_type_key == "video"
        grid_t = max(mm_count // image_processor.temporal_patch_size, 1)

    grid_h = resized_height // image_processor.patch_size
    grid_w = resized_width // image_processor.patch_size
    vision_tokens = grid_t * grid_h * grid_w
    llm_num_vision_tokens = (vision_tokens // image_processor.merge_size //
                             image_processor.merge_size)

    return resized_height, resized_width, llm_num_vision_tokens


class Qwen2VLImageInput(VLMInput):
    def __init__(self, model: str, tokenizer: Optional[str]):
        super().__init__(model, tokenizer)
        self.processor = AutoProcessor.from_pretrained(self.model)
        self.image_processor = self.processor.image_processor
        self.modality = "image"

    def get_stop_token_ids(self):
        return None

    def get_placeholder(self):
        image_token_id = self.hf_config.image_token_id
        image_place_holder = self.tokenizer.decode(image_token_id)
        return image_place_holder

    def get_demo_prompt(self, question: str, **kwargs):
        if kwargs["mm_per_prompt"] > 1:
            pass
        placeholders = [
            {"type": "image", "image": None}
            for _ in range(kwargs["mm_per_prompt"])
        ]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        if kwargs["mm_per_prompt"] > 1:
            if not os.path.isdir(input_vision_file):
                raise ValueError(
                    "input_vision_file must be a path when multi-image mode"
                )
            image_data = []
            curr_num = 0
            for file in os.listdir(input_vision_file):
                try:
                    image = Image.open(f"{input_vision_file}/{file}").convert(
                        "RGB"
                    )
                except Exception:
                    logging.warning(f"file read failed: {file}")
                    continue
                image_data.append(image)
                curr_num += 1
                if curr_num == kwargs["mm_per_prompt"]:
                    break
            if len(image_data) != kwargs["mm_per_prompt"]:
                raise ValueError(
                    "image in input_vision_file must equal to mm_per_prompt"
                )
        else:
            try:
                image_data = Image.open(input_vision_file).convert("RGB")
            except Exception as error:
                raise ValueError(f"file read failed: {input_vision_file}")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        vision_shapes = input_vision_shape.split(";")
        if kwargs["mm_per_prompt"] != len(vision_shapes):
            raise ValueError(
                "mm_per_prompt must be equal to input_vision_shape group"
            )

        image_feature_sizes = []
        for vision_shape in vision_shapes:
            input_shape = vision_shape.split(",")
            input_height = int(input_shape[-2])
            input_width = int(input_shape[-1])
            _, _, llm_num_vision_tokens = _get_vision_info(
                image_processor=self.image_processor,
                height=input_height,
                width=input_width,
                min_pixels=self.image_processor.min_pixels,
                max_pixels=self.image_processor.max_pixels,
                do_resize=self.image_processor.do_resize,
                data_type_key="image",
                mm_count=1,
            )
            image_feature_sizes.append(llm_num_vision_tokens)
        print(f'Image feature size {image_feature_sizes}')
        return image_feature_sizes

    def get_dummy_prompt(
        self,
        input_len: int,
        placeholder: str,
        image_feature_sizes: List[int],
        **kwargs,
    ):
        prompt = "hi" * input_len
        special_tokens_len = len(self.tokenizer(prompt).input_ids) - input_len
        if special_tokens_len > 0:
            prompt = "hi" * (input_len - special_tokens_len)
        if placeholder:
            for image_feature_size in image_feature_sizes:
                prompt = prompt.replace(
                    "hi" * image_feature_size,
                    placeholder,
                    1,
                )
        return prompt


class Qwen2VLVideoInput(VLMInput):
    def __init__(self, model: str, tokenizer: Optional[str]):
        super().__init__(model, tokenizer)
        self.processor = AutoProcessor.from_pretrained(self.model)
        self.image_processor = self.processor.image_processor
        self.modality = "video"

    def get_stop_token_ids(self):
        return None

    def get_placeholder(self):
        vision_start_token_id = self.hf_config.image_token_id
        vision_end_token_id = self.hf_config.image_token_id
        vision_token_id = self.hf_config.image_token_id

        return self.tokenizer.decode(
            [vision_start_token_id, vision_token_id, vision_end_token_id]
        )

    def get_demo_prompt(self, question: str, **kwargs):
        placeholders = [
            {"type": "video", "video": None}
            for _ in range(kwargs["mm_per_prompt"])
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": question},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        input_vision_file = input_vision_file.split(",")
        num_frames = kwargs.get("num_frames")

        videos = []
        for vision_info in input_vision_file:
            video = video_to_ndarrays(os.path.realpath(vision_info), num_frames=num_frames)
            videos.append(video)
        return {self.modality: videos}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        vision_shapes = input_vision_shape.split(";")
        if kwargs["mm_per_prompt"] != len(vision_shapes):
            raise ValueError(
                "mm_per_prompt must be equal to input_vision_shape group"
            )
        num_frames = kwargs.get("num_frames")
        image_feature_sizes = []
        for vision_shape in vision_shapes:
            input_shape = vision_shape.split(",")
            input_height = int(input_shape[-2])
            input_width = int(input_shape[-1])
            _, _, llm_num_vision_tokens = _get_vision_info(
                image_processor=self.image_processor,
                height=input_height,
                width=input_width,
                min_pixels=self.image_processor.min_pixels,
                max_pixels=self.image_processor.max_pixels,
                do_resize=self.image_processor.do_resize,
                data_type_key="image",
                mm_count=1,
            )
            image_feature_sizes.append(llm_num_vision_tokens * num_frames)
        return image_feature_sizes

    def get_dummy_prompt(
        self,
        input_len: int,
        placeholder: str,
        image_feature_sizes: List[int],
        **kwargs,
    ):
        prompt = "hi" * input_len
        special_tokens_len = len(self.tokenizer(prompt).input_ids) - input_len
        if special_tokens_len > 0:
            prompt = "hi" * (input_len - special_tokens_len)
        if placeholder:
            for image_feature_size in image_feature_sizes:
                prompt = prompt.replace(
                    "hi" * image_feature_size,
                    placeholder,
                    1,
                )
        return prompt

    def get_dummy_vision_data(self, input_vision_shape: str, **kwargs):
        input_shape = input_vision_shape.split(",")
        image_height = int(input_shape[-2])
        image_width = int(input_shape[-1])
        pil_image = Image.new("RGB", (image_width, image_height), color=0)
        np_frame = np.array(pil_image)
        mm_data_per_video = np.repeat([np_frame], kwargs["num_frames"], axis=0)
        mm_data = {self.modality: mm_data_per_video}
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
