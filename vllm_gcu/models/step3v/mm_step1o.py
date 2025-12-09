# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from itertools import product
from math import ceil, sqrt
from typing import Any, List, Literal, Optional, Tuple, TypedDict, Union, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, TensorType
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.pooler import Pooler,  PoolingType
from vllm.v1.outputs import SamplerOutput, PoolerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm_gcu.models.step3v.step_encoder import StepCLIPVisionModel
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
    MultiModalKwargsItems,
    NestedTensors,
    MultiModalDataDict,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer


from .sentencepiece_tokenizer import SentencePieceTokenizer

from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.utils import (
    flatten_bn,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    maybe_prefix,
    merge_multimodal_embeddings,
)


DEFAULT_HIGH_RESOLUTION = (
    os.getenv("VLLM_DEFAULT_HIGH_RESOLUTION", "false").lower() == "true"
)
print(f"DEFAULT_HIGH_RESOLUTION: {DEFAULT_HIGH_RESOLUTION}")

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]
"""
The output embeddings must be one of the following formats:

- A list or tuple of 2D tensors, where each tensor corresponds to
    each input multimodal data item (e.g, image).
- A single 3D tensor, with the batch dimension grouping the 2D tensors.
"""


class GPUToTensor(torch.nn.Module):

    def forward(
        self, raw_image: Union[np.ndarray, Image.Image]
    ) -> torch.Tensor:
        if isinstance(raw_image, Image.Image):
            return transforms.ToTensor()(raw_image)
        if raw_image.ndim == 2:
            raw_image = raw_image[:, :, None].repeat(3, -1)
        if torch.gcu.is_available():
            device = torch.device("gcu")
        else:
            device = torch.device("cpu")
        image_tensor = torch.from_numpy(raw_image).to(device)
        image_tensor = torch.permute(image_tensor, (2, 0, 1)).contiguous()
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32).div(255)
        return image_tensor


class StepPreprocessor:

    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = patch_size if patch_size is not None else size

        self.transform = transforms.Compose(
            [
                GPUToTensor(),
                transforms.Normalize(mean, std),
                transforms.Resize(
                    (size, size),
                    interpolation=(
                        InterpolationMode.BICUBIC
                        if interpolation_mode == "bicubic"
                        else InterpolationMode.BILINEAR
                    ),
                    antialias=True,
                ),
            ]
        )

        self.patch_transform = (
            transforms.Compose(
                [
                    GPUToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.Resize(
                        (patch_size, patch_size),
                        interpolation=(
                            InterpolationMode.BICUBIC
                            if interpolation_mode == "bicubic"
                            else InterpolationMode.BILINEAR
                        ),
                        antialias=True,
                    ),
                ]
            )
            if patch_size is not None
            else None
        )

    def preprocess(self, image, return_tensors="pt", is_patch=False):  # noqa
        if is_patch:
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}
        else:
            return {
                "pixel_values": self.transform(image).unsqueeze(0)
            }  # compatible with CLIPImageProcessor


class MMStep1oImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: (
        torch.Tensor
    )  # (batch_size * num_images, num_channels, height, width)
    patch_pixel_values: Optional[
        torch.Tensor
    ]  # (batch_size * num_patches, num_channels, patch_size, patch_size)
    num_patches: List[int]  # (batch_size * num_patches)


class MMStep1oImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: (
        torch.Tensor
    )  # (batch_size * num_images * image_feature_size, hidden_size)


MMStep1oImageInputs = Union[
    MMStep1oImagePixelInputs, MMStep1oImageEmbeddingInputs
]

ImageWithPatches = Tuple[Image.Image, list[Image.Image], list[int] | None]


class ImagePatcher:

    def determine_window_size(self, long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
        assert 1 >= img_rate_thr >= 0, "The `in_rate_thr` should lie in 0~1"
        windows = []
        # Sliding windows.
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step

            x_num = (
                1 if width <= size_w else ceil((width - size_w) / step_w + 1)
            )
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            y_num = (
                1 if height <= size_h else ceil((height - size_h) / step_h + 1)
            )
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)

        return [
            (
                int(box[0]),
                int(box[1]),
                int(box[2] - box[0]),
                int(box[3] - box[1]),
            )
            for box in windows
        ], (x_num, y_num)

    def square_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    def get_image_size_for_padding(
        self, img_width: int, img_height: int
    ) -> Tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(
        self, img_width: int, img_height: int
    ) -> Tuple[int, int]:

        if max(img_height, img_width) > 3024:
            scale_factor = 3024 / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
            return img_width, img_height
        else:
            return img_width, img_height

    def get_image_size_for_crop(
        self, img_width: int, img_height: int, window_size: int
    ):
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            xiaoshu_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if xiaoshu_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            xiaoshu_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if xiaoshu_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    def patch_crop(self, img: Image.Image, i: int, j: int, th: int, tw: int):
        target = img.crop((j, i, j + tw, i + th))
        return target

    def get_num_patches(
        self, img_width: int, img_height: int
    ) -> Tuple[int, int]:
        img_width, img_height = self.get_image_size_for_padding(
            img_width, img_height
        )
        img_width, img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        window_size = self.determine_window_size(
            max(img_height, img_width), min(img_height, img_width)
        )
        if window_size == 0:
            return 0, 0
        else:
            img_width, img_height = self.get_image_size_for_crop(
                img_width, img_height, window_size
            )
            center_list, (x_num, y_num) = self.slide_window(
                img_width,
                img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            full_rows = (len(center_list) - 1) // x_num + 1
            if len(center_list) > 0 and len(center_list) % x_num == 0:
                full_rows -= 1
            return len(center_list), full_rows

    def __call__(
        self, img: Image.Image
    ) -> Tuple[Image.Image, List[Image.Image], List[bool] | None]:
        img_width, img_height = img.size
        new_img_width, new_img_height = self.get_image_size_for_padding(
            img_width, img_height
        )
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = img.size

        new_img_width, new_img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        img = img.resize(
            (new_img_width, new_img_height), Image.Resampling.BILINEAR
        )
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width),
            min(new_img_height, new_img_width),
        )

        if window_size == 0:
            return img, [], None
        else:
            new_img_width, new_img_height = self.get_image_size_for_crop(
                new_img_width, new_img_height, window_size
            )
            if (new_img_width, new_img_height) != (img_width, img_height):
                img_for_crop = img.resize(
                    (new_img_width, new_img_height), Image.Resampling.BILINEAR
                )
            else:
                img_for_crop = img

            patches = []
            newlines = []
            center_list, (x_num, y_num) = self.slide_window(
                new_img_width,
                new_img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            for patch_id, center_lf_point in enumerate(center_list):
                x, y, patch_w, patch_h = center_lf_point
                big_patch = self.patch_crop(
                    img_for_crop, y, x, patch_h, patch_w
                )
                patches.append(big_patch)
                if (patch_id + 1) % x_num == 0:
                    newlines.append(patch_id)

            if newlines and newlines[-1] == len(patches) - 1:
                newlines.pop()

            return (
                img,
                patches,
                (
                    [i in newlines for i in range(len(patches))]
                    if len(patches) > 0
                    else None
                ),
            )


class Step1oProcessor:

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.image_size = 728
        self.patch_size = 504
        self.image_preprocessor = StepPreprocessor(
            self.image_size, "bilinear", self.patch_size
        )

        self.num_image_feature_size = 169
        self.num_patch_feature_size = 81
        self.image_token = "<im_patch>"
        self.image_feature_placeholder = (
            self.image_token * self.num_image_feature_size
        )
        self.patch_feature_placeholder = (
            self.image_token * self.num_patch_feature_size
        )

        self.patcher = ImagePatcher()

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.image_token]

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        num_patches, num_newlines = self.patcher.get_num_patches(
            img_width, img_height
        )
        return (
            num_patches * (self.num_patch_feature_size + 2)
            + self.num_image_feature_size
            + 2
            + num_newlines
        )

    def _split_images(
        self, images: list[Image.Image]
    ) -> list[ImageWithPatches]:
        result = []
        for img in images:
            detail = img.info.get("detail", None)
            use_high_resolution = False
            if detail == "high":
                use_high_resolution = True
            elif detail == "low":
                use_high_resolution = False
            else:
                use_high_resolution = DEFAULT_HIGH_RESOLUTION

            if use_high_resolution:
                result.append(self.patcher(img))
            else:
                result.append((img, [], None))
        return result

    def _convert_images_to_pixel_values(
        self,
        images: list[Image.Image],
        is_patch: bool = False,
    ) -> list[torch.Tensor]:
        return [
            self.image_preprocessor.preprocess(img, is_patch=is_patch)[
                "pixel_values"
            ]
            for img in images
        ]

    def _get_patch_repl(
        self,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> Tuple[str, list[int]]:
        text = ""
        token_ids = []
        for i in range(num_patches):
            assert len(patch_newline_mask) == num_patches
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            token_ids.extend(
                [self.tokenizer.convert_tokens_to_ids("<patch_start>")]
                + [self.image_token_id] * self.num_patch_feature_size
                + [self.tokenizer.convert_tokens_to_ids("<patch_end>")]
            )
            if patch_newline_mask and patch_newline_mask[i]:
                text += "<patch_newline>"
                token_ids.append(
                    self.tokenizer.convert_tokens_to_ids("<patch_newline>")
                )
        return text, token_ids

    def _get_image_repl(
        self,
        num_images: int,
    ) -> Tuple[str, list[int]]:
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"
        token_ids = (
            [self.tokenizer.convert_tokens_to_ids("<im_start>")]
            + [self.image_token_id] * self.num_image_feature_size
            + [self.tokenizer.convert_tokens_to_ids("<im_end>")]
        )
        return text * num_images, token_ids * num_images

    def _get_image_repl_features(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: Optional[list[bool]],
    ) -> Tuple[str, list[int]]:
        if num_patches > 0:
            patch_repl, patch_repl_ids = self._get_patch_repl(
                num_patches, patch_new_line_idx
            )
        else:
            patch_repl = ""
            patch_repl_ids = []
        image_repl, image_repl_ids = self._get_image_repl(num_images)
        return patch_repl + image_repl, patch_repl_ids + image_repl_ids

    def replace_placeholder(
        self, text: str, placeholder: str, repls: list[str]
    ) -> str:
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            image_inputs = {}
            if isinstance(self.tokenizer, SentencePieceTokenizer):
                assert len(text) == 1
                text_inputs = {
                    "input_ids": torch.tensor(
                        [
                            self.tokenizer.encode(
                                text[0], include_control_tokens=True
                            )
                        ],
                        dtype=torch.long,
                    )
                }  # step-tokenizer does not support text input for special tokens
            else:
                text_inputs = self.tokenizer(text)
        else:
            split_images_data = self._split_images(images)
            pixel_values_lst = []
            patch_pixel_values_lst = []
            patch_newline_mask_lst = []
            image_repl_str_lst = []
            image_repl_ids_lst = []
            num_patches = []
            for (
                raw_img,
                img_patches,
                patch_newline_mask,
            ) in split_images_data:
                pixel_values_lst.extend(
                    self._convert_images_to_pixel_values([raw_img])
                )

                if len(img_patches) > 0:
                    patch_pixel_values_lst.extend(
                        self._convert_images_to_pixel_values(
                            img_patches, is_patch=True
                        )
                    )
                num_patches.append(len(img_patches))

                image_repl_str, image_repl_ids = self._get_image_repl_features(
                    1, len(img_patches), patch_newline_mask
                )
                image_repl_str_lst.append(image_repl_str)
                image_repl_ids_lst.extend(image_repl_ids)

                if patch_newline_mask is not None:
                    patch_newline_mask_lst.extend(patch_newline_mask)

            image_inputs = {
                "pixel_values": torch.cat(pixel_values_lst),
                "num_patches": num_patches,
            }
            if patch_pixel_values_lst:
                image_inputs["patch_pixel_values"] = torch.cat(
                    patch_pixel_values_lst
                )
            if patch_newline_mask_lst:
                image_inputs["patch_newline_mask"] = torch.tensor(
                    patch_newline_mask_lst, dtype=torch.bool
                )

            if isinstance(self.tokenizer, SentencePieceTokenizer):
                text_inputs = {
                    "input_ids": torch.tensor(
                        image_repl_ids_lst, dtype=torch.long
                    ).unsqueeze(0)
                }  # step-tokenizer does not support text input for special tokens
            else:
                text = [
                    self.replace_placeholder(
                        t, self.image_token, image_repl_str_lst
                    )
                    for t in text
                ]
                text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


class Step1oProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self) -> Step1oProcessor:
        return Step1oProcessor(
            self.get_hf_config(),
            self.get_tokenizer(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_max_image_tokens(self) -> int:
        hf_processor = self.get_hf_processor()
        return hf_processor.get_num_image_tokens(
            self.get_image_size_with_most_features().width,
            self.get_image_size_with_most_features().height,
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(728, 728)


class Step1oDummyInputsBuilder(BaseDummyInputsBuilder[Step1oProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        target_width, target_height = (
            self.info.get_image_size_with_most_features()
        )
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            )
        }

        return ProcessorInputs(
            prompt="<im_patch>" * num_images,
            mm_data=mm_data,
        )

    # TODO: @abstractmethod after transition
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        Build the text input corresponding to :code:`mm_counts`.
        """
        if (
            type(self).get_dummy_processor_inputs
            == BaseDummyInputsBuilder.get_dummy_processor_inputs
        ):
            raise NotImplementedError

        # logging.warning_once("`get_dummy_processor_inputs` has been split up "
        #                     "into `get_dummy_text` and `get_dummy_mm_data`. "
        #                     "These two methods will be marked as abstract "
        #                     "in an upcoming release.")

        seq_len = self.info.ctx.model_config.max_model_len
        return self.get_dummy_processor_inputs(seq_len, mm_counts).prompt

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images),
        }


class Step1oMultiModalProcessor(BaseMultiModalProcessor[Step1oProcessingInfo]):

    #def _get_prompt_updates(
    #    self,
    #    mm_items: MultiModalDataItems,
    #    hf_processor_mm_kwargs: Mapping[str, Any],
    #    out_mm_kwargs: MultiModalKwargs,
    #) -> Sequence[PromptUpdate]:
    #    hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
    #    image_placeholder_token_id = hf_processor.image_token_id
    #    print(f"============== out_mm_kwargs: {out_mm_kwargs}")
    #    batch_num_patches = out_mm_kwargs["num_patches"].tolist()

    #    def get_replacement_step1o(item_idx: int):
    #        img_out = out_mm_kwargs.get_item("image", item_idx)
    #        num_patches = batch_num_patches[item_idx]
    #        if num_patches > 0:
    #            patch_newline_mask = img_out["patch_newline_mask"].data.tolist()
    #            image_repl_ids = hf_processor._get_image_repl_features(
    #                1, num_patches, patch_newline_mask
    #            )[1]
    #        else:
    #            image_repl_ids = hf_processor._get_image_repl_features(
    #                1, 0, None
    #            )[1]
    #        return image_repl_ids

    #    return [
    #        PromptReplacement(
    #            modality="image",
    #            target=[image_placeholder_token_id],
    #            replacement=get_replacement_step1o,
    #        )
    #    ]

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_placeholder_token_id = hf_processor.image_token_id

        def get_replacement_step1o(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            num_patches = int(out_item["num_patches"].data)
            if num_patches > 0:
                patch_newline_mask = out_item["patch_newline_mask"].data
                image_repl_ids = hf_processor._get_image_repl_features(
                    1, num_patches, patch_newline_mask.tolist()
                )[1]
            else:
                image_repl_ids = hf_processor._get_image_repl_features(1, 0, None)[1]
            return PromptUpdateDetails.select_token_id(
                seq=image_repl_ids,
                embed_token_id=image_placeholder_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[image_placeholder_token_id],
                replacement=get_replacement_step1o,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            patch_pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches
            ),
            num_patches=MultiModalFieldConfig.batched("image"),
            patch_newline_mask=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches
            ),
        )


@MULTIMODAL_REGISTRY.register_processor(
    Step1oMultiModalProcessor,
    info=Step1oProcessingInfo,
    dummy_inputs=Step1oDummyInputsBuilder,
)
class MMGPTStep1oForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_model = StepCLIPVisionModel(
            config.vision_tower_config,
            None,
            prefix=maybe_prefix(prefix, "vision_model"),
            need_dp=False,
        )
        self.vit_downsampler = nn.Conv2d(
            config.vision_tower_config.hidden_size,
            config.vision_tower_config.output_hidden_size,
            kernel_size=2,
            stride=config.understand_projector_stride,
        )
        self.vit_downsampler2 = nn.Conv2d(
            config.vision_tower_config.output_hidden_size,
            config.vision_tower_config.output_hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.vit_large_projector = nn.Linear(
            config.vision_tower_config.output_hidden_size * 2,
            config.hidden_size,
            bias=config.projector_bias,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return Sampler()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Optional[MMStep1oImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        patch_pixel_values = kwargs.pop("patch_pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = flatten_bn(pixel_values, concat=True)
            if pixel_values.dim() >= 3:
                pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:])
            if patch_pixel_values is not None:
                patch_pixel_values = flatten_bn(patch_pixel_values, concat=True)
                patch_pixel_values = patch_pixel_values.view(
                    -1, *patch_pixel_values.shape[-3:]
                )
                # Handle empty patch_pixel_values by setting to None
                if patch_pixel_values.shape[0] == 0:
                    patch_pixel_values = None
            num_patches = flatten_bn(num_patches, concat=True).tolist()

            return MMStep1oImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values.to(self.dtype).to(self.device),
                patch_pixel_values=(
                    patch_pixel_values.to(self.dtype).to(self.device)
                    if patch_pixel_values is not None
                    else None
                ),
                num_patches=num_patches,
            )

        if image_embeds is not None:
            if image_embeds.dim() == 2 or image_embeds.dim() >= 3:
                image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
            else:
                raise ValueError(
                    f"Unexpected shape for image_embeds: {image_embeds.shape}"
                )

            return MMStep1oImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds.to(self.dtype).to(self.device),
            )
        return None

    def _process_image_features(
        self, image_features: torch.Tensor
    ) -> torch.Tensor:
        B, P = image_features.shape[:2]
        HW = int(sqrt(P))
        image_features = image_features.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vit_downsampler(image_features)
        image_features = self.vit_downsampler2(image_features)
        n_dim = image_features.size(1)
        image_features = image_features.view(B, n_dim, -1).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features

    def _process_image_input(
        self, image_input: MMStep1oImageInputs
    ) -> tuple[torch.Tensor, ...]:

        if image_input["type"] == "image_embeds":
            image_features = image_input["image_embeds"]
        else:
            image_features = self.vision_model(image_input["pixel_values"])[0][
                :, 4:
            ]
            patch_image_features = (
                self.vision_model(image_input["patch_pixel_values"])[0][:, 4:]
                if image_input["patch_pixel_values"] is not None
                else None
            )
            num_patches = image_input["num_patches"]

        image_features = self._process_image_features(image_features)
        patch_image_features = (
            self._process_image_features(patch_image_features)
            if patch_image_features is not None
            else None
        )

        merged_image_features = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[
                    cur_patch_idx : cur_patch_idx + num_patch
                ]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(
                image_features[i].view(-1, image_features.shape[-1])
            )
            cur_patch_idx += num_patch
            merged_image_features.append(
                torch.cat(cur_feature)
                if len(cur_feature) > 1
                else cur_feature[0]
            )
        return merged_image_features

    def get_multimodal_embeddings(self, **kwargs) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        #vision_embeddings: Optional[NestedTensors] = None,
        multimodal_embeddings: MultiModalEmbeddings | None = None
    ) -> torch.Tensor:
        #if vision_embeddings is None:
        if multimodal_embeddings is None:
            inputs_embeds = self.language_model.model.get_input_embeddings(
                input_ids
            )
        else:
            is_text = input_ids != self.config.image_token_id
            text_ids = input_ids[is_text]
            text_embeds = self.language_model.model.get_input_embeddings(
                text_ids
            )
            inputs_embeds = torch.empty(
                input_ids.shape[0],
                text_embeds.shape[-1],
                dtype=text_embeds.dtype,
                device=text_embeds.device,
            )
            inputs_embeds[is_text] = text_embeds
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                #vision_embeddings,
                multimodal_embeddings,
                self.config.image_token_id,
            )
        return inputs_embeds

#    def get_input_embeddings(
#        self,
#        input_ids: torch.Tensor,
#        multimodal_embeddings: MultiModalEmbeddings | None = None,
#        *,
#        is_multimodal: torch.Tensor | None = None,
#        # Multi-modal token ID may exceed vocab size
#        handle_oov_mm_token: bool = True,
#    ) -> torch.Tensor:
#        # This is to satisfy the type checker for each overload
#        if multimodal_embeddings is None or is_multimodal is None:
#            return super().get_input_embeddings(input_ids)

#        return super().get_input_embeddings(
#            input_ids,
#            multimodal_embeddings=multimodal_embeddings,
#            is_multimodal=is_multimodal,
#            handle_oov_mm_token=handle_oov_mm_token,
#        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            # always pass the input via `inputs_embeds`
            # to make sure the computation graph is consistent
            inputs_embeds = self.get_input_embeddings(
                input_ids, vision_embeddings,
#                is_multimodal=input_ids == self.config.image_token_id,
            )
            input_ids = None
        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        #sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(
            hidden_states,
            #sampling_metadata
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def maybe_remap_params(self, name):
        if name.startswith("model."):
            name = name.replace("model.", "language_model.model.")
        if name.startswith("lm_head"):
            name = name.replace("lm_head", "language_model.lm_head")
        if name.startswith("vision_model."):
            name = name.replace("vision_model.", "vision_model.vision_model.")
        return name

    def load_weights_1o(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:

            name = self.maybe_remap_params(name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        params_need_to_load = []
        for name in params_dict:
            params_need_to_load.append(name)
        params_need_to_load = set(params_need_to_load)

        if params_need_to_load != loaded_params:
            param_name_example = list(params_need_to_load - loaded_params)[0]
            raise RuntimeError(
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"
            )

    def load_weights_3v(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        from vllm.model_executor.layers.fused_moe import FusedMoE

        qkv_params_mapping = [
            # (param_name, shard_name, relative_start_idx, relative_end_idx)
            (".qkv_proj", ".q_proj", 0, self.config.share_q_dim / (self.config.share_q_dim + self.config.head_dim * 2)),
            (".qkv_proj", ".k_proj", self.config.share_q_dim / (self.config.share_q_dim + self.config.head_dim * 2), (self.config.share_q_dim + self.config.head_dim) / (self.config.share_q_dim + self.config.head_dim * 2)),
            (".qkv_proj", ".v_proj", (self.config.share_q_dim + self.config.head_dim) / (self.config.share_q_dim + self.config.head_dim * 2), (self.config.share_q_dim + self.config.head_dim * 2) / (self.config.share_q_dim + self.config.head_dim * 2)),
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        params_need_to_load = set()

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.language_model.model.config.moe_num_experts,
        )

        if self.language_model.model.use_fused_moe:
            expert_params_mapping = [
                (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
                (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
                (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),
            ]
        else:
            expert_params_mapping = []

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        for name, loaded_weight in weights:
            name = self.maybe_remap_params(name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if any(
                    disable_moe_stacked_param in name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for expert_id in range(loaded_weight.shape[0]):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        start_idx,
                        end_idx,
                    ) in qkv_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(
                            param.output_dim, begin_idx, end_idx - begin_idx
                        )
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)

        params_need_to_load = []
        for name in params_dict:
            params_need_to_load.append(name)
        params_need_to_load = set(params_need_to_load)

        if params_need_to_load != loaded_params:
            param_name_example = list(params_need_to_load - loaded_params)[0]
            raise RuntimeError(
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if self.config.model_type in ["step1o", "mmgpt_qwen2_v2"]:
            self.load_weights_1o(weights)
        elif self.config.model_type == "step3v":
            self.load_weights_3v(weights)
        else:
            raise ValueError(
                f"Unsupported model type: {self.multimodal_config.model_type}"
            )


class MMGPTStep1oRewardModel(MMGPTStep1oForCausalLM, VllmModelForPooling):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        pooler_config = vllm_config.model_config.pooler_config
        assert (
            pooler_config is not None
        ), "Pooler config must be provided for classification models"

        # Remove attributes specific to CausalLM if they exist directly on self
        # (They are typically part of language_model)
        for attr in ("sampler", "lm_head"):
            if hasattr(self.language_model, attr):
                delattr(self.language_model, attr)

        # Initialize the classification score head
        self.score = RowParallelLinear(
            config.text_config.hidden_size,
            config.num_labels,  # Assumes num_labels is in the main config
            quant_config=quant_config,
            input_is_parallel=False,
            bias=False,
            prefix=maybe_prefix(prefix, "score"),
        )

        # Initialize the pooler
        # Use LAST pooling, no normalization, apply softmax (typical for classification)
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.ALL,
            normalize=False,
            softmax=False,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        return self._pooler(hidden_states, pooling_metadata)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Get hidden states from the base model (without the LM head)
        hidden_states = super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Apply the classification head
        logits, _ = self.score(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Filter out lm_head weights before passing to the base loader
        weights_iterator = (
            (name, data)
            for name, data in weights
            if "language_model.lm_head." not in name
        )

        # Use the base class's load_weights logic, which now includes
        # handling for the 'score' layer via maybe_remap_params
        super().load_weights(weights_iterator)
