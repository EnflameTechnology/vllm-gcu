import math
from typing import Iterable, Optional, Tuple

import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode
from transformers.activations import ACT2FN

from vllm.distributed import get_tensor_model_parallel_world_size

# from vllm.model_executor.layers.layernorm import OptimusLayerNorm
try:
    from vllm.model_executor.layers.layernorm import OptimusLayerNorm
except ImportError:
    from torch.nn.modules.normalization import LayerNorm as OptimusLayerNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from .step3v_config import CLIPVisionConfig


def get_abs_pos(abs_pos, tgt_size):
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = (
            old_pos_embed.view(1, src_size, src_size, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(
            1, tgt_size * tgt_size + 1, dim
        )
        return vision_pos_embed
    else:
        return abs_pos


class StepCLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.pad_tp_size = 4  # hard code for padding
        # To load the pretrained weights, we still use P+1 as the seqlen
        self.position_embedding = torch.nn.Embedding(
            self.num_patches + 1, self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches + 1).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # pad
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(self.position_ids), patch_embeds.size(1)
        )
        embeddings = torch.cat(
            [
                embeddings[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.pad_tp_size - 1, 1),
                embeddings,
            ],
            dim=1,
        )
        return embeddings


class StepCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        need_dp: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.total_num_heads

        self.scale = self.head_dim**-0.5

        if not need_dp:
            tp_size = get_tensor_model_parallel_world_size()
            assert self.total_num_heads % tp_size == 0
            self.num_heads = self.total_num_heads // tp_size
            self.qkv_proj = QKVParallelLinear(
                self.embed_dim,
                self.head_dim,
                self.total_num_heads,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )
            self.out_proj = RowParallelLinear(
                self.embed_dim,
                self.embed_dim,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )
        else:
            self.num_heads = self.total_num_heads
            self.qkv_proj = ReplicatedLinear(
                self.embed_dim,
                self.head_dim * 3,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )
            self.out_proj = ReplicatedLinear(
                self.embed_dim,
                self.embed_dim,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual=None,
        layernorm=None,
    ):
        """Input shape: Batch x Time x Channel"""
        if layernorm is not None:
            hidden_states = layernorm(hidden_states)

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim)
        if self.head_dim % 16 != 0 or (
            self.head_dim != 64 and self.head_dim != 128
        ):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, scale=self.scale, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).reshape(
                bsz, tgt_len, self.num_heads * self.head_dim
            )
        else:
            from optimus import flash_attn_func
            attn_output = flash_attn_func(
                q, k, v, softmax_scale=self.scale, causal=False
            )
            attn_output = attn_output.view(
                bsz, tgt_len, self.num_heads * self.head_dim
            )

        # attn_output, _ = self.out_proj(attn_output, residual=residual)
        if residual is not None:
            attn_output += residual
        attn_output, _ = self.out_proj(attn_output)

        return attn_output


class StepCLIPMLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        need_dp: bool = False,
    ):
        super().__init__()
        self.config = config
        # self.activation_fn = torch.compile(ACT2FN[config.hidden_act])
        self.activation_fn = ACT2FN[config.hidden_act]
        if not need_dp:
            self.fc1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )
            self.fc2 = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )
        else:
            self.fc1 = ReplicatedLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )
            self.fc2 = ReplicatedLinear(
                config.intermediate_size,
                config.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=prefix,
            )

    def forward(
        self, hidden_states: torch.Tensor, residual=None, layernorm=None
    ) -> torch.Tensor:
        if layernorm is not None:
            hidden_states = layernorm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        # hidden_states, _ = self.fc2(hidden_states, residual=residual)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class StepCLIPEncoderLayer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        need_dp: bool = False,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = StepCLIPAttention(
            config, quant_config, prefix=f"{prefix}.self_attn", need_dp=need_dp
        )
        self.layer_norm1 = OptimusLayerNorm(
            self.embed_dim, eps=config.layer_norm_eps
        )
        self.mlp = StepCLIPMLP(
            config, quant_config, prefix=f"{prefix}.mlp", need_dp=need_dp
        )
        self.layer_norm2 = OptimusLayerNorm(
            self.embed_dim, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        residual = self.layer_norm1(
            self.self_attn(
                hidden_states=hidden_states, residual=None, layernorm=None
            )
        )
        h = hidden_states + residual
        out = h + self.layer_norm2(self.mlp(h))
        return out


class StepCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        need_dp: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                StepCLIPEncoderLayer(
                    config,
                    quant_config,
                    prefix=f"{prefix}.layers.{i}",
                    need_dp=need_dp,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
            )
        return hidden_states


class StepCLIPVisionTransformer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        need_dp: bool = False,
    ):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.vision_model_preprocessor = torchvision.transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        self.embeddings = StepCLIPVisionEmbeddings(config)
        self.transformer = StepCLIPEncoder(
            config,
            quant_config,
            prefix=f"{prefix}.transformer",
            need_dp=need_dp,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.transformer(inputs_embeds=hidden_states)
        return hidden_states, None


class StepCLIPVisionModel(nn.Module):
    _PARAMS_KEYS_TO_SELECT = ["vision_model"]

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        need_dp: bool = False,
    ):
        super().__init__()
        if quant_config is not None:
            quant_config = (
                None  # FIXME(ys): step encoder does not support quantization
            )
        self.vision_model = StepCLIPVisionTransformer(
            config,
            quant_config,
            prefix=f"{prefix}.vision_model",
            need_dp=need_dp,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        return self.vision_model(pixel_values=pixel_values)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        _params_to_ignore = [
            "text_model",
            "logit_scale",
            "vision_model.embeddings.position_ids",
            "visual_projection.weight",
            "text_projection.weight",
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            if any(param_name in name for param_name in _params_to_ignore):
                continue

            if not (
                any(
                    param_name in name
                    for param_name in self._PARAMS_KEYS_TO_SELECT
                )
            ):
                continue
            if name.startswith("model.vision_tower.vision_tower"):
                name = name.replace("model.vision_tower.vision_tower.", "")
            elif name.startswith("model.vision_tower"):
                name = name.replace("model.vision_tower.", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name.split("."):
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

        params_need_to_load = set(params_dict.keys())
        if params_need_to_load != loaded_params:
            param_name_example = list(params_need_to_load - loaded_params)[0]
            raise RuntimeError(
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"
            )


class StepCLIPVisionModelWithPostprocess(StepCLIPVisionModel):
    _PARAMS_KEYS_TO_SELECT = ["vision_model", "vit_downsampler"]

    def __init__(self, config: CLIPVisionConfig, need_dp: bool = True):
        super().__init__(config, need_dp)
        self.config = config
        self.vit_downsampler = nn.Conv2d(
            self.config.hidden_size,
            self.config.output_hidden_size,
            kernel_size=2,
            stride=2,
        )
        self.vit_downsampler2 = nn.Conv2d(
            self.config.output_hidden_size,
            self.config.output_hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        x = super().forward(x)[0][:, 4:]
        B, P = x.shape[:2]
        HW = int(math.sqrt(P))
        x = x.permute(0, 2, 1).view(B, self.config.hidden_size, HW, HW)
        x = self.vit_downsampler(x)
        x = self.vit_downsampler2(x)
        x = x.view(B, self.config.output_hidden_size * 2, -1).permute(0, 2, 1)
        return x
