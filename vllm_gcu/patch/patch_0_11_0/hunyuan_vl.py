from unittest.mock import patch
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.v1.worker.gpu_input_batch import CachedRequestState as OriginalCachedRequestState

from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding, DeepseekScalingRotaryEmbedding, DualChunkRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding, DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding, Llama3RotaryEmbedding, Llama4VisionRotaryEmbedding,
    MRotaryEmbedding, NTKScalingRotaryEmbedding, Phi3LongRoPEScaledRotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from vllm_gcu.models.hunyuan_vl.xdrope import XDRotaryEmbedding


@dataclass
class CachedRequestState(OriginalCachedRequestState):
    xdrope_positions: Optional[torch.Tensor] = None


def uses_xdrope_dim(self) -> int:
    """Detect if the model with this config uses XD-ROPE."""
    config = self.hf_config
    xdrope_section = getattr(config, "xdrope_section", None)
    if xdrope_section is not None and isinstance(xdrope_section, list):
        return len(xdrope_section)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return 0

    if isinstance(rope_scaling, dict) and "xdrope_section" in rope_scaling:
        xdrope_section = rope_scaling["xdrope_section"]
        if xdrope_section is not None and isinstance(xdrope_section, list):
            return len(xdrope_section)

    return 0


_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}

def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: Optional[dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    dual_chunk_attention_config: Optional[dict[str, Any]] = None,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           rope_scaling_args, dual_chunk_attention_args, dtype)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if dual_chunk_attention_config is not None:
        extra_kwargs = {
            k: v
            for k, v in dual_chunk_attention_config.items()
            if k in ("chunk_size", "local_size")
        }
        rotary_emb = DualChunkRotaryEmbedding(head_size, rotary_dim,
                                              max_position, base,
                                              is_neox_style, dtype,
                                              **extra_kwargs)
    elif not rope_scaling:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                                     is_neox_style, dtype)
    else:
        scaling_type = rope_scaling["rope_type"]

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            rotary_emb = Llama3RotaryEmbedding(head_size, rotary_dim,
                                               max_position, base,
                                               is_neox_style, dtype,
                                               scaling_factor, low_freq_factor,
                                               high_freq_factor,
                                               original_max_position)
        elif scaling_type == "mllama4":
            rotary_emb = Llama4VisionRotaryEmbedding(head_size, rotary_dim,
                                                     max_position, base,
                                                     is_neox_style, dtype)
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved",
                                                       False),
                )
            else:
                rotary_emb = RotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                )
        elif scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim,
                                                      max_position, base,
                                                      is_neox_style,
                                                      scaling_factor, dtype)
        elif scaling_type == "ntk":
            scaling_factor = rope_scaling["factor"]
            mixed_b = rope_scaling.get('mixed_b', None)
            rotary_emb = NTKScalingRotaryEmbedding(head_size, rotary_dim,
                                                   max_position, base,
                                                   is_neox_style,
                                                   scaling_factor, dtype,
                                                   mixed_b)
        elif scaling_type == "dynamic":
            if "alpha" in rope_scaling:
                scaling_alpha = rope_scaling["alpha"]
                rotary_emb = DynamicNTKAlphaRotaryEmbedding(
                    head_size, rotary_dim, max_position, base, is_neox_style,
                    scaling_alpha, dtype)
            elif "factor" in rope_scaling:
                scaling_factor = rope_scaling["factor"]
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    head_size, rotary_dim, max_position, base, is_neox_style,
                    scaling_factor, dtype)
            else:
                raise ValueError("Dynamic rope scaling must contain either "
                                 "'alpha' or 'factor' field")
        elif scaling_type == "xdrope":
            scaling_alpha = rope_scaling["alpha"]
            rotary_emb = XDRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_alpha,
                dtype,
                xdrope_section=rope_scaling["xdrope_section"],
            )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow")
            }
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    original_max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved",
                                                       False),
                    scaling_factor=scaling_factor,
                    **extra_kwargs)
            else:
                rotary_emb = YaRNScalingRotaryEmbedding(
                    head_size, rotary_dim, original_max_position, base,
                    is_neox_style, scaling_factor, dtype, **extra_kwargs)
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            # assert max_position == original_max_position * scaling_factor
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size, rotary_dim, original_max_position, base,
                is_neox_style, scaling_factor, dtype, **extra_kwargs)
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size, rotary_dim, max_position, original_max_position,
                base, is_neox_style, dtype, short_factor, long_factor,
                **extra_kwargs)
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


patch("vllm.model_executor.layers.rotary_embedding.get_rope", get_rope).start()
patch("vllm.config.model.ModelConfig.uses_xdrope_dim", property(uses_xdrope_dim), create=True).start()
patch("vllm.v1.worker.gpu_input_batch.CachedRequestState", CachedRequestState).start()


import vllm.model_executor.models.hunyuan_v1 as hunyuan_v1
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (xd, seq_len) if xdrope is enabled for hunyuan-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HunYuanModel(hunyuan_v1.HunYuanModel):
    pass


patch("vllm.model_executor.models.hunyuan_v1.HunYuanModel", HunYuanModel).start()