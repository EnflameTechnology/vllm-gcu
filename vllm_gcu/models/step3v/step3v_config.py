# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Optional, Union

from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig

from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig


class StepConfig(PretrainedConfig):
    model_type = "step"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        moe_every_n_layer: int = 2,  # 2 means 50% layers use MoE, interleaved with normal non-MoE layers.
        use_moe: bool = False,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        max_pos_interp_ratio: float = 1,
        alibi_slopes: Optional[List[float]] = None,
        moe_layer_offset: int = 0,
        moe_dynamic_exp_p: float = 1.0,
        rope_theta: float = 500000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embedding: int = 16384,
        share_expert_dim: Optional[int] = None,
        allgather_dtype: Optional[str] = None,
        share_q_dim: Optional[int] = None,
        norm_expert_weight: bool = True,
        bos_token_id: Optional[Union[List[int], int]] = None,
        eos_token_id: Optional[Union[List[int], int]] = None,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_every_n_layer = moe_every_n_layer
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.max_pos_interp_ratio = max_pos_interp_ratio
        self.alibi_slopes = alibi_slopes
        self.moe_layer_offset = moe_layer_offset
        self.moe_dynamic_exp_p = moe_dynamic_exp_p

        # for step2 mini
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        self.max_position_embedding = max_position_embedding
        if share_expert_dim is None:
            self.share_expert_dim = self.moe_intermediate_size * self.moe_top_k
        else:
            self.share_expert_dim = share_expert_dim
        self.share_q_dim = share_q_dim
        self.norm_expert_weight = norm_expert_weight

        self.allgather_dtype = allgather_dtype
        self._verify_slopes()

        super().__init__(
            bos_token_id=1 if bos_token_id is None else bos_token_id,
            eos_token_id=[2, 3] if eos_token_id is None else eos_token_id,
            **kwargs,
        )

    def _verify_slopes(self):
        if self.alibi_slopes is None:
            return
        if len(self.alibi_slopes) != self.num_attention_heads:
            raise ValueError(
                f"Number of alibi_slopes ({len(self.alibi_slopes)}) does not match num_attention_heads ({self.num_attention_heads})"
            )


class Step1Config(StepConfig):
    model_type = "step1"


class Step2Config(StepConfig):
    model_type = "step2"

    def __init__(self, use_offline_input_scales: bool = True, **kwargs):
        self.use_offline_input_scales = use_offline_input_scales
        super().__init__(**kwargs)


class Step2MiniConfig(StepConfig):
    model_type = "step2_mini"


class CLIPVisionConfig(PretrainedConfig):
    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class SamViTConfig(PretrainedConfig):
    model_type = "sam_vit_model"

    def __init__(
        self,
        depth=24,
        embed_dim=1024,
        image_size=1280,
        mlp_ratio=4,
        num_heads=16,
        patch_size=16,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        global_attn_indexes=(5, 11, 17, 23),
        window_size=14,
        out_channels=256,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.global_attn_indexes = global_attn_indexes
        self.window_size = window_size
        self.out_channels = out_channels
        self.layer_norm_eps = layer_norm_eps


class MMGPTStep1Config(Step1Config):
    # for step1.5
    model_type = "mmgpt_step1"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        use_im_start_end=True,
        vision_select_layer=-2,
        image_token_len=None,
        projector_stride=1,
        vision_tower_config=None,
        image_token_id=13,
        image_seq_length=400,
        bos_token_id: Optional[Union[List[int], int]] = None,
        eos_token_id: Optional[Union[List[int], int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=1 if bos_token_id is None else bos_token_id,
            eos_token_id=[2, 3] if eos_token_id is None else eos_token_id,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        self.image_token_len = image_token_len
        self.projector_stride = projector_stride
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.vision_tower_config = (
            CLIPVisionConfig(**vision_tower_config)
            if vision_tower_config is not None
            else None
        )
        self.text_config = Step1Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_attention_groups=num_attention_groups,
            num_hidden_layers=num_hidden_layers,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            architectures=["Step1ForCausalLM"],
            torch_dtype=getattr(self, "torch_dtype", "bfloat16"),
        )


class MMGPTStep1ConfigV2(Step1Config):
    # for step1.5c/step1u, models with both vit and sam encoders
    model_type = "mmgpt_step1_v2"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        use_im_start_end=True,
        vision_select_layer=-1,
        image_token_len=None,
        understand_projector_stride=1,
        vit_scale=1.0,
        projector_bias=True,
        vision_tower_config=None,
        sam_model_config=None,
        image_token_id=13,
        bos_token_id: Optional[Union[List[int], int]] = None,
        eos_token_id: Optional[Union[List[int], int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=1 if bos_token_id is None else bos_token_id,
            eos_token_id=[2, 3] if eos_token_id is None else eos_token_id,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        self.image_token_len = image_token_len
        self.image_token_id = image_token_id
        self.understand_projector_stride = understand_projector_stride
        self.vit_scale = vit_scale
        self.projector_bias = projector_bias
        self.vision_tower_config = (
            CLIPVisionConfig(**vision_tower_config)
            if vision_tower_config is not None
            else None
        )
        self.sam_model_config = (
            SamViTConfig(**sam_model_config)
            if sam_model_config is not None
            else None
        )
        self.text_config = Step1Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_attention_groups=num_attention_groups,
            num_hidden_layers=num_hidden_layers,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            architectures=["Step1ForCausalLM"],
            torch_dtype=getattr(self, "torch_dtype", "bfloat16"),
        )


class Step1oConfig(Step1Config):
    # for step1o
    model_type = "step1o"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        use_im_start_end=True,
        vision_select_layer=-1,
        image_token_len=None,
        image_token_id=13,
        understand_projector_stride=1,
        vit_scale=1.0,
        projector_bias=True,
        patch_token_len=None,
        vision_tower_config=None,
        bos_token_id: Optional[Union[List[int], int]] = None,
        eos_token_id: Optional[Union[List[int], int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=1 if bos_token_id is None else bos_token_id,
            eos_token_id=[2, 3] if eos_token_id is None else eos_token_id,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        self.image_token_len = image_token_len
        self.image_token_id = image_token_id
        self.understand_projector_stride = understand_projector_stride
        self.vit_scale = vit_scale
        self.projector_bias = projector_bias
        self.patch_token_len = (
            patch_token_len
            if patch_token_len is not None
            else self.image_token_len
        )
        self.vision_tower_config = (
            CLIPVisionConfig(**vision_tower_config)
            if vision_tower_config is not None
            else None
        )
        self.text_config = Step1Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_attention_groups=num_attention_groups,
            num_hidden_layers=num_hidden_layers,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            architectures=["Step1ForCausalLM"],
            torch_dtype=getattr(self, "torch_dtype", "bfloat16"),
        )


class MMGPTQwen2Config(PretrainedConfig):
    # for step1.5t
    model_type = "mmgpt_qwen2"

    def __init__(
        self,
        vocab_size=64012,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_attention_groups=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_im_start_end=True,
        vision_select_layer=-1,
        image_token_len=None,
        image_token_id=151656,
        understand_projector_stride=1,
        vit_scale=1.0,
        projector_bias=True,
        pad_token_id=-1,
        vision_tower_config=None,
        sam_model_config=None,
        eos_token_id=None,
        **kwargs,
    ) -> None:
        if eos_token_id is not None:
            if isinstance(eos_token_id, list):
                eos_token_id = list(set([151643, 151646] + eos_token_id))
            else:
                eos_token_id = [151643, 151646, eos_token_id]
        else:
            eos_token_id = [151643, 151646]

        super().__init__(eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        self.image_token_len = image_token_len
        self.image_token_id = image_token_id
        self.understand_projector_stride = understand_projector_stride
        self.vit_scale = vit_scale
        self.projector_bias = projector_bias
        self.pad_token_id = pad_token_id
        self.vision_tower_config = (
            CLIPVisionConfig(**vision_tower_config)
            if vision_tower_config is not None
            else None
        )
        self.sam_model_config = (
            SamViTConfig(**sam_model_config)
            if sam_model_config is not None
            else None
        )

        self.text_config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            architectures=["Qwen2ForCausalLM"],
            torch_dtype=getattr(self, "torch_dtype", "bfloat16"),
        )


class MMGPTQwen2ConfigV2(MMGPTQwen2Config):
    model_type = "mmgpt_qwen2_v2"

    def __init__(
        self,
        vocab_size=64012,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_attention_groups=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_im_start_end=True,
        vision_select_layer=-1,
        image_token_len=None,
        image_token_id=151675,
        understand_projector_stride=1,
        vit_scale=1.0,
        projector_bias=True,
        pad_token_id=-1,
        vision_tower_config=None,
        sam_model_config=None,
        eos_token_id=None,
        **kwargs,
    ) -> None:
        if eos_token_id is not None:
            if isinstance(eos_token_id, list):
                eos_token_id = list(
                    set([151643, 151645, 151665] + eos_token_id)
                )
            else:
                eos_token_id = [151643, 151645, 151665, eos_token_id]
        else:
            eos_token_id = [151643, 151645, 151665]

        super().__init__(eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        self.image_token_len = image_token_len
        self.image_token_id = image_token_id
        self.understand_projector_stride = understand_projector_stride
        self.vit_scale = vit_scale
        self.projector_bias = projector_bias
        self.pad_token_id = pad_token_id
        self.vision_tower_config = (
            CLIPVisionConfig(**vision_tower_config)
            if vision_tower_config is not None
            else None
        )
        self.sam_model_config = (
            SamViTConfig(**sam_model_config)
            if sam_model_config is not None
            else None
        )

        self.text_config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            architectures=["Qwen2ForCausalLM"],
            torch_dtype=getattr(self, "torch_dtype", "bfloat16"),
        )


class Step3vConfig(Step1Config):
    model_type = "step3v"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        moe_every_n_layer: int = 2,  # 2 means 50% layers use MoE, interleaved with normal non-MoE layers.
        use_moe: bool = False,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        max_pos_interp_ratio: float = 1,
        alibi_slopes: Optional[List[float]] = None,
        moe_layer_offset: int = 0,
        moe_dynamic_exp_p: float = 1.0,
        rope_theta: float = 500000,
        rope_scaling: Optional[dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embedding: int = 16384,
        share_expert_dim: Optional[int] = None,
        allgather_dtype: Optional[str] = None,
        share_q_dim: Optional[int] = None,
        norm_expert_weight: bool = True,
        moe_layers_enum: Optional[str] = None,
        use_im_start_end: bool = True,
        vision_select_layer: int = -1,
        image_token_len: Optional[int] = None,
        image_token_id: int = 128001,
        understand_projector_stride: int = 1,
        vit_scale: float = 1.0,
        projector_bias: bool = True,
        patch_token_len: Optional[int] = None,
        vision_tower_config: Optional[dict[str, Any]] = None,
        bos_token_id: Optional[Union[List[int], int]] = None,
        eos_token_id: Optional[Union[List[int], int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=0 if bos_token_id is None else bos_token_id,
            eos_token_id=[1, 128805] if eos_token_id is None else eos_token_id,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.moe_every_n_layer = moe_every_n_layer
        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.max_pos_interp_ratio = max_pos_interp_ratio
        self.alibi_slopes = alibi_slopes
        self.moe_layer_offset = moe_layer_offset
        self.moe_dynamic_exp_p = moe_dynamic_exp_p
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        self.max_position_embedding = max_position_embedding
        self.share_expert_dim = share_expert_dim
        self.allgather_dtype = allgather_dtype
        self.share_q_dim = share_q_dim
        self.norm_expert_weight = norm_expert_weight
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        self.image_token_len = image_token_len
        self.image_token_id = image_token_id
        self.understand_projector_stride = understand_projector_stride
        self.vit_scale = vit_scale
        self.projector_bias = projector_bias
        self.patch_token_len = (
            patch_token_len
            if patch_token_len is not None
            else self.image_token_len
        )
        self.vision_tower_config = (
            CLIPVisionConfig(**vision_tower_config)
            if vision_tower_config is not None
            else None
        )
        self.text_config = Step2MiniConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_attention_groups=num_attention_groups,
            num_hidden_layers=num_hidden_layers,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            moe_every_n_layer=moe_every_n_layer,
            use_moe=use_moe,
            moe_intermediate_size=moe_intermediate_size,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            max_pos_interp_ratio=max_pos_interp_ratio,
            alibi_slopes=alibi_slopes,
            moe_layer_offset=moe_layer_offset,
            moe_dynamic_exp_p=moe_dynamic_exp_p,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embedding=max_position_embedding,
            share_expert_dim=share_expert_dim,
            allgather_dtype=allgather_dtype,
            share_q_dim=share_q_dim,
            norm_expert_weight=norm_expert_weight,
            moe_layers_enum=moe_layers_enum,
            architectures=["Step2MiniForCausalLM"],
            torch_dtype=getattr(self, "torch_dtype", "bfloat16"),
        )
