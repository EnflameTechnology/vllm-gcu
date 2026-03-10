from functools import lru_cache
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch
import torch_gcu

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLVisionConfig,
    Qwen3_VisionTransformer,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_gcu.models.qwen2_5_vl import Qwen2_5_VisionAttention


class Qwen3_VisionTransformerModify(Qwen3_VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        with patch('vllm.model_executor.models.qwen3_vl.Qwen2_5_VisionAttention', Qwen2_5_VisionAttention):
            super().__init__(vision_config=vision_config, 
                             norm_eps=norm_eps,
                             quant_config=quant_config,
                             prefix=prefix,
                             use_data_parallel=use_data_parallel)

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        # Support both Tensor and list inputs for DP path
        if isinstance(grid_thw, list):
            grid_list = grid_thw
            max_grid_size = max(max(h, w) for _, h, w in grid_list)
        else:
            grid_list = grid_thw.tolist()
            max_grid_size = int(grid_thw[:, 1:].max().item())
        
        pos_ids = [
            self.rot_pos_ids(h, w, self.spatial_merge_size)
            if t == 1
            else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)

        rotary_pos_emb_full = self.rotary_pos_emb.get_cos_sin(max_grid_size)
        cos, sin = rotary_pos_emb_full.chunk(2, dim=-1)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        rotary_pos_emb = torch.cat([cos_combined, sin_combined], -1)

        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw):
        patch_pos_embeds_permute = []
        num_grid = self.num_grid_per_side * self.num_grid_per_side

        embeds = torch.arange(num_grid, device=self.pos_embed.weight.device)
        embeds = (
            self.pos_embed(embeds)
            .permute(1, 0)
            .reshape(1, -1, self.num_grid_per_side, self.num_grid_per_side)
        )
        for t, h, w in grid_thw:
            pos_embed = torch.nn.functional.interpolate(
                embeds, size=(h, w), mode="bilinear", align_corners=False
            )
            pos_embed = pos_embed.reshape(
                -1,
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            pos_embed = pos_embed.permute(1, 3, 2, 4, 0)
            pos_embed = pos_embed.flatten(0, 3).repeat(t, 1)
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)
    
    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.numpy()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw_list)

        cu_seqlens_np = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype=np.int32
        )
        cu_seqlens_np  = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens_np])
        cu_seqlens = torch.empty(len(cu_seqlens_np), dtype=torch.int32, pin_memory=True)
        cu_seqlens.copy_(torch.from_numpy(cu_seqlens_np))
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        hidden_states = hidden_states.unsqueeze(1)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens_np)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                rotary_pos_emb=rotary_pos_emb,
                                max_seqlen=max_seqlen,
                                seqlens=seqlens)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(
                    layer_num)
                deepstack_feature = self.deepstack_merger_list[
                    deepstack_merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists,
            dim=1)  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states


patch("vllm.model_executor.models.qwen3_vl.Qwen3_VisionTransformer", Qwen3_VisionTransformerModify).start()

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
