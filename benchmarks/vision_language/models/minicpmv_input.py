import os
import numpy as np
from typing import Optional, List, Union
from decord import VideoReader, cpu
from PIL import Image
from vllm_utils.vision_language.models.base import VLMInput
from vllm.model_executor.models.minicpmv import input_processor_for_minicpmv
from vllm.inputs.registry import InputContext
from vllm.config import ModelConfig


class MinicpmvImageInput(VLMInput):

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

        from vllm.multimodal.image import cached_get_image_processor
        image_processor = cached_get_image_processor(self.model,
                                                     trust_remote_code=True)

        max_slice_nums = 1 if len(vision_shapes) > 16 else 2
        image_feature_sizes = []
        for i in range(len(vision_shapes)):
            input_shape = vision_shapes[i].split(",")
            input_height = int(input_shape[-2])
            input_width = int(input_shape[-1])

            placeholder = image_processor. \
            get_slice_image_placeholder(
                (input_height, input_width),
                i,
                use_image_id=False,
                max_slice_nums=max_slice_nums
            )

            image_feature_sizes.append(len(self.tokenizer.encode(placeholder)))

        return image_feature_sizes

    def get_dummy_prompt(self, input_len: int, placeholder: str,
                         image_feature_size: int,
                         **kwargs):

        mm_per_prompt = kwargs["mm_per_prompt"] if kwargs[
            "mm_per_prompt"] else 1

        input_vision_shape = kwargs["input_vision_shape"]
        kwargs.pop("input_vision_shape")

        model_config = ModelConfig(
            model=self.model,
            tokenizer=self.model,
            tokenizer_mode="auto",
            trust_remote_code=True,
            seed=0,
            dtype="auto",
            revision=None,
            limit_mm_per_prompt={"image": mm_per_prompt},
        )
        prompt = self.get_demo_prompt("hi", **kwargs)
        vision_data = self.get_dummy_vision_data(input_vision_shape, **kwargs)

        input = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": vision_data["image"]
            }
        }
        data = input_processor_for_minicpmv(InputContext(model_config), input)
        fake_prompt_len = input_len - len(data["prompt_token_ids"])
        fake_prompt = "hi" * fake_prompt_len
        all_prompt = prompt + fake_prompt
        return all_prompt

    def get_stop_token_ids(self):
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens
        ]
        return stop_token_ids
   
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
        image_token_id = 151646
        image_place_holder = self.tokenizer.decode(image_token_id)
        return image_place_holder


class MinicpmvVideoInput(MinicpmvImageInput):

    def __init__(self, model: str, tokenizer: Optional[str]):
        super().__init__(model, tokenizer)
        self.modality = "video"

    def get_demo_prompt(self, question, **kwargs):
        num_frame = kwargs["num_frames"] if "num_frames" in kwargs.keys(
        ) else 8

        # demo
        messages = [{
            'role':
            'user',
            "content":
            "".join(["(<image>./</image>)"] * num_frame) + "\n" + question
        }]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def get_demo_vision_data(self, input_vision_file: str, **kwargs):
        num_frame = kwargs["num_frames"] if kwargs["num_frames"] else 8

        def encode_video(filepath):

            def uniform_sample(l, n):
                gap = len(l) / n
                idxs = [int(i * gap + gap / 2) for i in range(n)]
                return [l[i] for i in idxs]

            vr = VideoReader(filepath, ctx=cpu(0))
            sample_fps = round(vr.get_avg_fps() / 1)  # FPS
            frame_idx = [i for i in range(0, len(vr), sample_fps)]
            if len(frame_idx) > num_frame:
                frame_idx = uniform_sample(frame_idx, num_frame)
            video = vr.get_batch(frame_idx).asnumpy()
            video = [Image.fromarray(v.astype('uint8')) for v in video]
            return video

        image_data = encode_video(input_vision_file)
        return {
            "image": {
                "images": image_data,
                "use_image_id": False,
                "max_slice_nums": 1 if num_frame > 16 else 2
            }
        }

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
            vision_data = {"image": {"images":images,
                        "use_image_id":False,
                        "max_slice_nums": 1 if len(images) > 16 else 2}}
            return prompt, vision_data
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}.")

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

        mm_data = {"image":
        {
            "images": pil_images,
            "use_image_id": False,
            "max_slice_nums": 1 if kwargs["mm_per_prompt"] > 16 else 2   
        }
        }
        return mm_data