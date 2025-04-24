# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

"""Inference-only GOT-OCR2 model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping, Sequence
from typing import (Any, Callable, Literal, Optional, Set, Tuple, TypedDict,
                    Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLImageProcessor

from vllm.config import VllmConfig
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalFieldConfig, MultiModalKwargs)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.transformers_utils.processor import (
    cached_image_processor_from_config)

from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from copy import deepcopy

from vllm.model_executor.models.interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from vllm.model_executor.models.utils import (AutoWeightsLoader, merge_multimodal_embeddings)

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from transformers.models.got_ocr2.processing_got_ocr2 import (
    GotOcr2Processor)
from transformers.models.got_ocr2 import GotOcr2Config, GotOcr2VisionConfig
from vllm.model_executor.models.utils import (AutoWeightsLoader, flatten_bn, merge_multimodal_embeddings)
from collections.abc import Iterable, Mapping, Sequence
from typing import (Any, Callable, Literal, Optional, Set, Tuple, TypedDict,
                    Union)
import os

class GotOcr2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size, num_channels, height, width)`"""

class GotOcr2MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


class GotOcr2VisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size):
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        return outputs


class GotOcr2VisionLayer(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GotOcr2VisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GotOcr2MLPBlock(config)
        self.window_size = window_size

    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


@dataclass
class GotOcr2VisionEncoderOutput(ModelOutput):
    """
    Base class for got_ocr2 vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class GotOcr2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        pixel_values = pixel_values.squeeze(1)
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class GotOcr2LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GotOcr2VisionNeck(nn.Module):
    def __init__(self, config: GotOcr2VisionConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = GotOcr2LayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = GotOcr2LayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class GotOcr2VisionEncoder(nn.Module):
    def __init__(self, config: GotOcr2VisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = GotOcr2PatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = GotOcr2VisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = GotOcr2VisionNeck(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GotOcr2VisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return GotOcr2VisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GotOcr2MultiModalProjector(nn.Module):
    def __init__(self, config: GotOcr2Config):
        super().__init__()
        vision_output_channels = config.vision_config.output_channels
        language_hidden_size = config.text_config.hidden_size
        self.conv_upsampler1 = nn.Conv2d(
            vision_output_channels, vision_output_channels * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv_upsampler2 = nn.Conv2d(
            vision_output_channels * 2, language_hidden_size, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.multimodal_projector = nn.Linear(language_hidden_size, language_hidden_size)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv_upsampler1(vision_embeddings)
        hidden_state = self.conv_upsampler2(hidden_state)
        hidden_state = hidden_state.flatten(2).permute(0, 2, 1)
        hidden_state = self.multimodal_projector(hidden_state)
        return hidden_state


class GotOcr2PreTrainedModel(PreTrainedModel):
    config_class = GotOcr2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GotOcr2VisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        # important: this ported version of GotOcr2 isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/GotOcr2/tree/main/got_ocr2 should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GotOcr2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(GotOcr2Config)

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> GotOcr2Processor:
        return self.ctx.get_hf_processor(
            GotOcr2Processor,
            image_processor=self.get_image_processor(min_pixels=min_pixels,
                                                     max_pixels=max_pixels,
                                                     size=size),
            **kwargs,
        )

    def _get_image_processor_kwargs(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ):
        if self.ctx.model_config.mm_processor_kwargs:
            kwargs.update(self.ctx.model_config.mm_processor_kwargs)

        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels

            if size is None:
                size = {"shortest_edge": min_pixels}
            else:
                size["shortest_edge"] = min_pixels

        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels

            if size is None:
                size = {"longest_edge": max_pixels}
            else:
                size["longest_edge"] = max_pixels

        if size is not None:
            kwargs["size"] = size

        return kwargs

    def get_image_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ):
        return cached_image_processor_from_config(
            self.ctx.model_config,
            **self._get_image_processor_kwargs(min_pixels=min_pixels,
                                               max_pixels=max_pixels,
                                               size=size,
                                               **kwargs),
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens()
        }

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> int:
        num_image_tokens = self.get_max_image_tokens()
        return num_image_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size = ImageSize(width=self.ctx.model_config.hf_image_processor_config['size']['width'],
                                   height=self.ctx.model_config.hf_image_processor_config['size']['height'])
        return max_image_size

    def get_max_image_tokens(self) -> int:
        return 256

def _got_ocr2_field_config(
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
    return dict(pixel_values=MultiModalFieldConfig.batched("image"))

class GotOcr2DummyInputsBuilder(BaseDummyInputsBuilder[GotOcr2ProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.img_pad_token

        target_width, target_height = 1024, 1024

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }
        return ProcessorInputs(
            prompt_text=image_token * 256,
            mm_data=mm_data,
        )

class GotOcr2MultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                fields_factory=_got_ocr2_field_config,
            )

        return super()._parse_image_data(data)

class GotOcr2MultiModalProcessor(BaseMultiModalProcessor[GotOcr2ProcessingInfo]
                                 ):

    def _get_data_parser(self) -> MultiModalDataParser:
        return GotOcr2MultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            tokenizer = self.info.get_tokenizer()
            prompt = [prompt] if isinstance(prompt, str) else prompt
            text_inputs = tokenizer(prompt,padding=False, return_tensors='pt')
            return BatchFeature(data={**text_inputs})
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            self.info._get_image_processor_kwargs(**mm_kwargs),
        )

    def _apply_hf_processor_text_only(self, prompt_text: str) -> list[int]:
        """
        Apply the HF processor on the prompt text only.

        Since HF processor requires that text and multi-modal items
        correspond to each other, we create dummy multi-modal items
        to go along with the text.
        """
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt_text)
        return prompt_ids


    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        img_start_token = vocab[hf_processor.img_start_token]
        img_pad_token = vocab[hf_processor.img_pad_token]
        img_end_token = vocab[hf_processor.img_end_token]

        def get_replacement(item_idx: int):

            num_image_tokens = self.info.get_num_image_tokens(image_width=1024,image_height=1024,image_processor=hf_processor)
            image_tokens = [img_pad_token] * num_image_tokens

            return image_tokens
        replacement_tokens = get_replacement(item_idx=0)
        return [
            PromptReplacement(
                modality="image",
                target=replacement_tokens,
                replacement=get_replacement,
            ),
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _got_ocr2_field_config(hf_inputs,hf_processor_mm_kwargs)

@MULTIMODAL_REGISTRY.register_processor(GotOcr2MultiModalProcessor,
                                        info=GotOcr2ProcessingInfo,
                                        dummy_inputs=GotOcr2DummyInputsBuilder)
class GotOcr2ForConditionalGeneration(GotOcr2PreTrainedModel, GenerationMixin, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = GotOcr2Config(**vars(vllm_config.model_config))
        super().__init__(config)
        self.config = config
        self.vision_tower = GotOcr2VisionEncoder(config.vision_config)

        self.multi_modal_projector = GotOcr2MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        vllm_config_c = deepcopy(vllm_config)
        vllm_config_c.model_config.hf_config=vllm_config_c.model_config.hf_config.text_config

        self.language_model = Qwen2ForCausalLM(vllm_config=vllm_config_c)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_index])
        return inputs_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values).last_hidden_state
        return self.multi_modal_projector(image_outputs)

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[GotOcr2ImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return GotOcr2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        return None
    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input_dict = self._parse_and_validate_image_input(**kwargs)
        image_input = image_input_dict['data'].to(self.dtype)
        # image_input = kwargs.pop("pixel_values", None)
        if image_input is None:
            return None
        vision_embeddings = self.get_image_features(image_input)

        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        pixel_values = kwargs.pop("pixel_values", None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values=pixel_values.to(inputs_embeds.dtype))
            n_image_tokens = (input_ids == self.config.image_token_index).sum()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)