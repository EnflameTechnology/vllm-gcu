import os
import numpy as np
from typing import Optional, List, Union
from decord import VideoReader, cpu
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput
from transformers import AutoProcessor


class MinicpmvImageInput(VLMInput):
    def __init__(self, model: str, tokenizer: Optional[str]):
        super().__init__(model, tokenizer)
        self.processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        self.image_processor = self.processor.image_processor

    def get_demo_prompt(self, question: str or List, **kwargs):
        mm_per_prompt = kwargs[
            "mm_per_prompt"] if "mm_per_prompt" in kwargs.keys() else 1
        # demo
        messages = [{
            'role':
            'user',
            "content":
            "".join(["(<image>./</image>)"] * mm_per_prompt) + "\n" + question
        }]

        prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        mm_per_prompt = kwargs["mm_per_prompt"] if kwargs[
            "mm_per_prompt"] else 1

        if kwargs["mm_per_prompt"] > 1:
            import os
            if os.path.isdir(input_vision_file):
                # multi image
                image_data = []
                for file in os.listdir(input_vision_file):
                    _, extend_type = os.path.splitext(file)
                    if extend_type in [".jpg", ".jpeg"]:
                        image = Image.open(
                            os.path.join(input_vision_file,
                                         file)).convert("RGB")
                        image_data.append(image)
                        if len(image_data) == kwargs["mm_per_prompt"]:
                            break
                if len(image_data) != kwargs["mm_per_prompt"]:
                    raise ValueError(
                        "image in input_vision_file must equal to mm_per_prompt"
                    )
                return {"image": image_data}
        else:
            # single image
            image_data = Image.open(input_vision_file).convert("RGB")
            return {"image": image_data}

    def get_image_feature_size(self, input_vision_shape: str, **kwargs):
        vision_shapes = input_vision_shape.split(";")
        if kwargs["mm_per_prompt"] != len(vision_shapes):
            raise ValueError(
                'mm_per_prompt must be equal to input_vision_shape group')

        image_feature_sizes = []
        for i in range(len(vision_shapes)):
            input_shape = vision_shapes[i].split(",")
            input_height = int(input_shape[-2])
            input_width = int(input_shape[-1])

            placeholder = self.image_processor.get_slice_image_placeholder(
                (input_height, input_width),
                i,
                )

            image_feature_sizes.append(len(self.tokenizer.encode(placeholder)))

        return image_feature_sizes

    def get_dummy_prompt(self, input_len: int, placeholder: str,
                         image_feature_sizes:  List[int],
                         **kwargs):

        mm_per_prompt = kwargs["mm_per_prompt"] if kwargs[
            "mm_per_prompt"] else 1

        prompt = self.get_demo_prompt("", **kwargs)
        prompt_len = len(self.tokenizer.encode(prompt)) - (len(self.tokenizer.encode("(<image>./</image>)"))-1) * mm_per_prompt

        total_image_feature_size = sum(image_feature_sizes)
        dummy_token_len = input_len - (prompt_len + total_image_feature_size)

        question = "hi" * (dummy_token_len)
        prompt = self.get_demo_prompt(question, **kwargs)

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
        return "(<image>./</image>)"

    def dataset_pred_post_process(self,
                                  dataset_name: str,
                                  pred: str):
        parsed_pred = pred.split("\n")[0].strip()
        all_choices = ["A", "B", "C", "D"]
        for choice in all_choices:
            if '{})'.format(choice) in parsed_pred:
                parsed_pred = choice
        return parsed_pred

class MinicpmvVideoInput(MinicpmvImageInput):

    def __init__(self, model: str, tokenizer: Optional[str]):
        super().__init__(model, tokenizer)
        self.modality = "video"

    def get_demo_prompt(self, question : str, **kwargs):
        # demo
        messages = [{
            'role': 'user',
            "content": "".join(["(<video>./</video>)"]) + "\n" + question
        }]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def dataset_pre_process(self,
                            dataset_name: str,
                            question: Union[str, List],
                            **kwargs):
        if dataset_name == "Video-MME":
            # dataset
            message = question
            content = []
            for x in message:
                if x['type'] == 'text':
                    content.append(x['value'])
                elif x['type'] == 'image':
                    image = Image.open(x['value']).convert('RGB')
                    content.append(image)
            msg = {'role': 'user', 'content': content}
            images = []
            content = msg["content"]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)
            messages = [msg]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            vision_data = {"image": images}
            return prompt, vision_data
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}.")

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        from vllm.assets.video import VideoAsset

        video = VideoAsset(
            name=input_vision_file,
                           num_frames=kwargs["num_frames"]).np_ndarrays

        return {self.modality: video}

    def get_placeholder(self):
        return "(<video>./</video>)"
