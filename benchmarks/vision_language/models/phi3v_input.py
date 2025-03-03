import os
import json
from typing import List
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput
from vllm.model_executor.models.phi3v import _calc_hd_transform_size


class Phi3VImageInput(VLMInput):
    def get_chat_template(self):
        template = "<|user|>\n{placeholder}{question}<|end|>\n<|assistant|>\n"
        return template

    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        # using tokenizer.apply_chat_template() produces same output, but too slow
        prompt = None
        placeholder = ""
        for i in range(kwargs["mm_per_prompt"]):
            placeholder += f"<|image_{i+1}|>\n"
        template = self.get_chat_template()
        prompt =  template.format(placeholder=placeholder, question=question)
        return prompt

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        if kwargs["mm_per_prompt"] > 1:
            if not os.path.isdir(input_vision_file):
                raise ValueError("input_vision_file must be a path when multi-image mode")
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
        # adapted from vllm.model_executor.models.phi3v.get_phi3v_image_feature_size
        vision_shapes = input_vision_shape.split(";")
        if kwargs["mm_per_prompt"] != len(vision_shapes):
            raise ValueError('mm_per_prompt must be equal to input_vision_shape group')

        image_feature_sizes = []
        for vision_shape in vision_shapes:
            input_shape = vision_shape.split(",")
            input_height = int(input_shape[-2])
            input_width = int(input_shape[-1])

            # get num_crops from model config
            preprocessor_config_filename = "preprocessor_config.json"
            preprocessor_config_path = os.path.join(self.model, preprocessor_config_filename)

            try:
                with open(preprocessor_config_path, 'r', encoding='utf-8') as file:
                    preprocessor_config_data = json.load(file)
            except FileNotFoundError:
                print(f"Error: The file '{preprocessor_config_path}' does not exist.")
            except Exception as e:
                print(f"Unexpected error: {e}")

            num_crops = preprocessor_config_data["num_crops"]

            # calculate image_feature_size
            new_width, new_height = _calc_hd_transform_size(width=input_width,
                                                            height=input_height,
                                                            hd_num=num_crops)
            image_feature_size = (new_height // 336 * new_width // 336 + 1) * 144 + 1 \
                                    + (new_height // 336 + 1) * 12
            image_feature_sizes.append(image_feature_size)
        return image_feature_sizes
    
    def get_dummy_prompt(self,
                         input_len: int,
                         placeholder: str,
                         image_feature_sizes: List[int],
                         **kwargs):
        base_prompt_len = len(self.tokenizer.encode(self.get_demo_prompt("", **kwargs)))
        multi_img_placeholder = ""
        for i in range(kwargs["mm_per_prompt"]):
            multi_img_placeholder += f"<|image_{i+1}|>\n"
        
        multi_img_placeholder_len = len(self.tokenizer.encode(multi_img_placeholder)) - 1 - kwargs["mm_per_prompt"]

        total_image_feature_size = sum(image_feature_sizes)
        dummy_token_len = input_len - (base_prompt_len + total_image_feature_size - multi_img_placeholder_len)

        question = "hi" * (dummy_token_len)
        prompt = self.get_demo_prompt(question, **kwargs)
        calculated_final_input_token_size = (len(self.tokenizer.encode(prompt)) + total_image_feature_size - multi_img_placeholder_len)

        assert calculated_final_input_token_size <= input_len, "Providing too many image tokens! Consider using smaller 'mm_per_prompt' or larger 'input_len'."
        assert calculated_final_input_token_size == input_len, "Expected number of input tokens does not equal to 'input_len'!"

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
        return "<|image|>"

    def dataset_pred_post_process(self,
                                  dataset_name: str,
                                  pred: str):
        parsed_pred = pred.strip()
        return parsed_pred



class Phi3VVideoInput(VLMInput):
    # keep self.modelity="image" because phi3v process videos in the same way as processing multi-images
    def get_chat_template(self):
        template = "<|user|>\n{placeholder}{question}<|end|>\n<|assistant|>\n"
        return template

    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        # using tokenizer.apply_chat_template() produces same output, but too slow
        prompt = None
        placeholder = ""
        for i in range(kwargs["mm_per_prompt"]):
            placeholder += f"<|image_{i+1}|>\n"
        template = self.get_chat_template()
        prompt =  template.format(placeholder=placeholder, question=question)
        return prompt

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        from vllm.assets.video import video_to_pil_images_list
        video_data = video_to_pil_images_list(path=input_vision_file,
                                num_frames=kwargs["mm_per_prompt"])

        return {self.modality: video_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        # adapted from vllm.model_executor.models.phi3v.get_phi3v_image_feature_size
        assert len(input_vision_shape.split(";")) == 1, "Phi3v only support one input dummy shape in video benchmark"
        
        input_shape = input_vision_shape.split(",")
        input_height = int(input_shape[-2])
        input_width = int(input_shape[-1])

        # get num_crops from model config
        preprocessor_config_filename = "preprocessor_config.json"
        preprocessor_config_path = os.path.join(self.model, preprocessor_config_filename)

        try:
            with open(preprocessor_config_path, 'r', encoding='utf-8') as file:
                preprocessor_config_data = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{preprocessor_config_path}' does not exist.")
        except Exception as e:
            print(f"Unexpected error: {e}")

        num_crops = preprocessor_config_data["num_crops"]

        # calculate image_feature_size
        new_width, new_height = _calc_hd_transform_size(width=input_width,
                                                        height=input_height,
                                                        hd_num=num_crops)
        image_feature_size = (new_height // 336 * new_width // 336 + 1) * 144 + 1 \
                                + (new_height // 336 + 1) * 12
        return image_feature_size
    
    def get_dummy_prompt(self,
                         input_len: int,
                         placeholder: str,
                         image_feature_size: int,
                         **kwargs):
        base_prompt_len = len(self.tokenizer.encode(self.get_demo_prompt("", **kwargs)))
        video_placeholder = ""
        for i in range(kwargs["mm_per_prompt"]):
            video_placeholder += f"<|image_{i+1}|>\n"
        
        video_placeholder_len = len(self.tokenizer.encode(video_placeholder)) - 1 - kwargs["mm_per_prompt"]

        total_image_feature_size = kwargs["mm_per_prompt"] * image_feature_size
        dummy_token_len = input_len - (base_prompt_len + total_image_feature_size - video_placeholder_len)

        question = "hi" * (dummy_token_len)
        prompt = self.get_demo_prompt(question, **kwargs)
        calculated_final_input_token_size = (len(self.tokenizer.encode(prompt)) + total_image_feature_size - video_placeholder_len)

        assert calculated_final_input_token_size <= input_len, "Providing too many image tokens! Consider using smaller 'mm_per_prompt' or larger 'input_len'."
        assert calculated_final_input_token_size == input_len, "Expected number of input tokens does not equal to 'input_len'!"

        return prompt
    
    def get_dummy_vision_data(self,
                              input_vision_shape: str,
                              **kwargs):
        assert len(input_vision_shape.split(";")) == 1, "Phi3v only support one input dummy shape in video benchmark"
        
        input_shape = input_vision_shape.split(",")
        image_height = int(input_shape[-2])
        image_width = int(input_shape[-1])
        pil_images = []

        for i in range(kwargs["mm_per_prompt"]):
            pil_image = Image.new("RGB", (image_width, image_height), color=0)
            pil_images.append(pil_image)

        mm_data = {self.modality: pil_images 
                   if kwargs["mm_per_prompt"] > 1 else pil_images[0]}
        return mm_data

    def get_placeholder(self):
        return "<|image|>"