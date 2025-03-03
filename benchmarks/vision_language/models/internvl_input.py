import os
import numpy as np
from typing import List, Optional, Union
from decord import VideoReader, cpu
from vllm_utils.vision_language.models.base import VLMInput
from PIL import Image
from vllm.model_executor.models.internvl import calculate_num_blocks,get_internvl_num_patches


class InternVLChatInput(VLMInput):
    def get_chat_template(self):
        template = "You are a helpful language and vision assistant." \
            "You are able to understand the visual content that the user provides," \
            " and assist the user with a variety of tasks using natural language.\n\nUser: {}\n\nAssistant:"
        return template

    def get_demo_prompt(self,
                        question: str):
        question = self.tokenizer.decode(92546) + question
        template = self.get_chat_template()
        prompt = template.format(question)
        return prompt

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        image_data = Image.open(input_vision_file).convert("RGB")
        return {self.modality: image_data}
    
    def _get_clip_patch_grid_length(self, image_size: int, patch_size: int) -> int:
        assert image_size % patch_size == 0
        return image_size // patch_size


    def _get_clip_num_patches(self, image_size: int, patch_size: int) -> int:
        grid_length = self._get_clip_patch_grid_length(image_size=image_size,
                                                patch_size=patch_size)
        return grid_length * grid_length

    def _get_internvl_num_patches(self, image_size: int, patch_size: int,
                             downsample_ratio: float):
        return int(
            self._get_clip_num_patches(image_size=image_size, patch_size=patch_size) *
            (downsample_ratio**2))

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        pass
    
class InternVL2Input(VLMInput):
    def get_demo_prompt(self,
                        question: str,
                        **kwargs):
        if kwargs["mm_per_prompt"] > 1:
            mm_per_prompt = kwargs["mm_per_prompt"] 
            placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i,_ in enumerate(range(mm_per_prompt), start=1))
            messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]
        else:
            messages = [{'role': 'user', 'content': f"<image>\n{question}"}]

        prompt = self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)
        return prompt

    def get_stop_token_ids(self):
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        return stop_token_ids

    def get_demo_vision_data(self,
                             input_vision_file: str,
                             **kwargs):
        if kwargs["mm_per_prompt"] > 1:
            import os
            if not os.path.isdir(input_vision_file):
                raise ValueError("input_vision_file must be a path when multi-image mode")
            image_data = []
            curr_num = 0
            for file in os.listdir(input_vision_file):
                _,extend_type = os.path.splitext(file)
                if extend_type == ".jpg":
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
        vision_shapes = input_vision_shape.split(";")
        if kwargs["mm_per_prompt"] != len(vision_shapes):
            raise ValueError('mm_per_prompt must be equal to input_vision_shape group')

        image_feature_sizes = []
        for vision_shape in vision_shapes:
            input_shape = vision_shape.split(",")
            input_height = int(input_shape[-2])
            input_width = int(input_shape[-1])
            vision_config = self.hf_config.vision_config
            downsample_ratio = self.hf_config.downsample_ratio

            image_size =vision_config.image_size
            patch_size = vision_config.patch_size
            num_patches = get_internvl_num_patches(image_size, patch_size,
                                                downsample_ratio)
            min_num = self.hf_config.min_dynamic_patch
            max_num = self.hf_config.max_dynamic_patch
            use_thumbnail = self.hf_config.use_thumbnail
            
            num_blocks, _, _ = calculate_num_blocks(input_width, input_height, min_num,
                                                    max_num, image_size,
                                                    use_thumbnail)
            image_feature_size = num_blocks * num_patches
            image_feature_sizes.append(image_feature_size)
        return image_feature_sizes

    def get_placeholder(self):
        image_token_id = 92546
        image_place_holder = self.tokenizer.decode(image_token_id)
        return image_place_holder
    
    def get_dummy_prompt(self,
                         input_len: int,
                         placeholder: str,
                         image_feature_sizes: List[int],
                         **kwargs):
        import re
        IMG_START = '<img>'
        IMG_END = '</img>'
        IMG_CONTEXT = '<IMG_CONTEXT>'
        new_prompt = self.get_demo_prompt("hi", **kwargs)
        image_idx = sorted(map(int, re.findall(r"Image-(\d+): <image>\n", new_prompt)))
        for idx, feature_size in enumerate(image_feature_sizes, start=1):
            image_prompt = IMG_START + IMG_CONTEXT * feature_size + IMG_END
            if not image_idx:
                image_prompt = f"Image-{idx}: {image_prompt}"
            new_prompt = new_prompt.replace('<image>', image_prompt, 1)
        new_prompt_token_ids = self.tokenizer.encode(new_prompt)
        fake_prompt_len = input_len - len(new_prompt_token_ids)
        fake_prompt = "hi" * fake_prompt_len
        special_tokens_len = len(self.tokenizer(new_prompt + fake_prompt).input_ids) - input_len
        if special_tokens_len > 0:
            fake_prompt = "hi" * (input_len - special_tokens_len)
        all_prompt = new_prompt + fake_prompt
        return all_prompt
    
    def dataset_pred_post_process(self,
                                  dataset_name: str,
                                  pred: str):
        parsed_pred = pred.split("<|im_end|>")[0].strip()
        all_choices = ["A", "B", "C", "D"]
        for choice in all_choices:
            if f'({choice})' in parsed_pred:
                parsed_pred = choice
        return parsed_pred
    
    


class InternVL2VideoInput(InternVL2Input):

    def __init__(self, model: str, tokenizer: Optional[str]):
        super().__init__(model, tokenizer)
        self.modality = "video"

    def get_demo_prompt(self, question, **kwargs):
        num_frame = kwargs["num_frames"] if "num_frames" in kwargs.keys(
        ) else 8

        # demo
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frame)])
        question = video_prefix + question
        
        messages = [{'role': 'user', 'content': f"{question}"}]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        num_frame = kwargs["num_frames"] if kwargs["num_frames"] else 8

        def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
            if bound:
                start, end = bound[0], bound[1]
            else:
                start, end = -100000, 100000
            start_idx = max(first_idx, round(start * fps))
            end_idx = min(round(end * fps), max_frame)
            seg_size = float(end_idx - start_idx) / num_segments
            frame_indices = np.array([
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ])
            return frame_indices

        def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
            # num_segments is the same meaning as num_frame
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())
            
            images = []

            pixel_values_list, num_patches_list = [], []
            
            frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
                images.append(img)

            return images

        # video processing reference: https://huggingface.co/OpenGVLab/InternVL2_5-2B
        image_data = load_video(input_vision_file, num_segments=num_frame)
        return {"image": image_data}

    def get_dummy_vision_data(self,
                              input_vision_shape: str,
                              **kwargs):
        vision_shapes = input_vision_shape.split(";")
        if kwargs["num_frames"] != len(vision_shapes):
            raise ValueError('num_frames must be equal to input_vision_shape group')

        pil_images = []
        for vision_shape in vision_shapes:
            input_shape = vision_shape.split(",")
            image_height = int(input_shape[-2])
            image_width = int(input_shape[-1])
            pil_image = Image.new("RGB", (image_width, image_height), color=0)
            pil_images.append(pil_image)

        mm_data = {"image":pil_images}
        return mm_data