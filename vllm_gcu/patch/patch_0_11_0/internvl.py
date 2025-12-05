#!/usr/bin/env python
# coding=utf-8
"""
Patch for vllm.model_executor.models.internvl.video_to_pixel_values_internvl
This patch fixes the TypeError when processing video frames with int64 dtype.
Reference: vllm_gcu/patch/patch_0_8_0/flash_attn.py
"""
from unittest.mock import patch
import torch
import numpy.typing as npt
import numpy as np
from PIL import Image

# Import the internvl module to ensure it's loaded
import vllm.model_executor.models.internvl

from vllm.model_executor.models.internvl import (
    get_internvl_target_ratios,
    build_transform,
    dynamic_preprocess_internvl,
)


def video_to_pixel_values_internvl_modify(
    video: npt.NDArray,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    frames_list = list[Image.Image]()
    for frame in video:
        pil_frame = dynamic_preprocess_internvl(
            Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), mode="RGB"),
            target_ratios=target_ratios,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
        )
        assert len(pil_frame) == 1
        frames_list.extend(pil_frame)

    pixel_values = torch.stack([transform(image) for image in frames_list])
    return pixel_values


# Apply patch using the same mechanism as vllm_gcu
# Reference: vllm_gcu/patch/patch_0_8_0/flash_attn.py
patcher = patch("vllm.model_executor.models.internvl.video_to_pixel_values_internvl", video_to_pixel_values_internvl_modify).start()