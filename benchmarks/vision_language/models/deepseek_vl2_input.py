import os
import math
import json
from typing import List
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput


class DeepSeekVL2Input(VLMInput):
    def get_chat_template(self):
        template = "<|User|>: {placeholder}{question}\n\n<|Assistant|>:"
        return template

    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        placeholder = "".join(f"image_{i}:<image>\n" 
                    for i in range(kwargs["mm_per_prompt"]))
        template = self.get_chat_template()
        prompt =  template.format(placeholder=placeholder, question=question)
        return prompt

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        if kwargs["mm_per_prompt"] > 1:
            if not os.path.isdir(input_vision_file):
                raise ValueError("input_vision_file must be a path in multi-image mode")
            image_data = []
            curr_num = 0
            for file in os.listdir(input_vision_file):
                _ , extend_type = os.path.splitext(file)
                if extend_type in [".jpg", ".png", ".bmp", ".webp"]:
                    image = Image.open(
                        os.path.join(input_vision_file, file)).convert("RGB")
                    image_data.append(image)
                    curr_num += 1
                    if curr_num == kwargs["mm_per_prompt"]:
                        break
            if len(image_data) != kwargs["mm_per_prompt"]:
                raise ValueError("image in input_vision_file must equal to mm_per_prompt")
        else:
            image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        # adapted from vllm.model_executor.models.deepseek_vl2.get_num_image_tokens

        def select_best_resolution(image_size, candidate_resolutions):
            # adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/faf18023f24b962b32d9f0a2d89e402a8d383a78/deepseek_vl2/models/processing_deepseek_vl_v2.py
            original_width, original_height = image_size
            best_fit = None
            max_effective_resolution = 0
            min_wasted_resolution = float("inf")

            for width, height in candidate_resolutions:
                scale = min(width / original_width, height / original_height)
                downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
                effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
                wasted_resolution = (width * height) - effective_resolution

                if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
                    max_effective_resolution = effective_resolution
                    min_wasted_resolution = wasted_resolution
                    best_fit = (width, height)

            return best_fit

        vision_shapes = input_vision_shape.split(";")
        if kwargs["mm_per_prompt"] != len(vision_shapes):
            raise ValueError('mm_per_prompt must be equal to input_vision_shape group')

        image_feature_sizes = []
        for vision_shape in vision_shapes:
            input_shape = vision_shape.split(",")
            input_height = int(input_shape[-2])
            input_width = int(input_shape[-1])

            # get processor parameters from model config
            processor_config_filename = "processor_config.json"
            processor_config_path = os.path.join(self.model, processor_config_filename)

            try:
                with open(processor_config_path, 'r', encoding='utf-8') as file:
                    processor_config_data = json.load(file)
            except FileNotFoundError:
                print(f"Error: The file '{processor_config_path}' does not exist.")
            except Exception as e:
                print(f"Unexpected error: {e}")

            patch_size = processor_config_data["patch_size"]
            downsample_ratio = processor_config_data["downsample_ratio"]
            image_size = 384

            # calculate image_feature_size
            candidate_resolutions = self.hf_config.candidate_resolutions
            best_width, best_height = select_best_resolution((input_width, input_height), candidate_resolutions)

            num_width_tiles, num_height_tiles = (best_width // image_size,
                                                best_height // image_size)
            h = w = math.ceil((image_size // patch_size) / downsample_ratio)

            global_views_tokens = h * (w + 1)
            local_views_tokens = (num_height_tiles * h) * (num_width_tiles * w + 1)
            total_tokens = global_views_tokens + local_views_tokens + 1
            image_feature_sizes.append(total_tokens)
        return image_feature_sizes
    
    def get_dummy_prompt(self,
                         input_len: int,
                         placeholder: str,
                         image_feature_sizes: List[int],
                         **kwargs):
        base_prompt_len = len(self.tokenizer.encode(self.get_demo_prompt("", **kwargs)))

        total_image_feature_size = sum(image_feature_sizes)
        dummy_token_len = input_len - (base_prompt_len + total_image_feature_size - kwargs["mm_per_prompt"]) - 1
        question = "hi" * (dummy_token_len)
        prompt = self.get_demo_prompt(question, **kwargs)
        calculated_final_input_token_size = len(self.tokenizer.encode(prompt)) - kwargs["mm_per_prompt"] + total_image_feature_size

        assert calculated_final_input_token_size <= input_len, "Providing too many image tokens! Consider using smaller 'mm_per_prompt' or larger 'input_len'."
        assert calculated_final_input_token_size == input_len, "Expected number of input tokens does not equal to 'input_len'!" # self check
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

    def get_placeholder(self):
        return "<image>"

    def dataset_pred_post_process(self,
                                  dataset_name: str,
                                  pred: str):
        parsed_pred = pred.strip()
        return parsed_pred