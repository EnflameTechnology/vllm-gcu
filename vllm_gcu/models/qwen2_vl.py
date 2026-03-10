import torch
import torch_gcu
import torch.nn.functional as F

from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from vllm.model_executor.models.qwen2_vl import Qwen2VisionTransformer
from unittest.mock import patch


class Qwen2VisionTransformerModify(Qwen2VisionTransformer):

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # compute cu_seqlens
        grid_thw_ = torch.tensor(grid_thw).to(self.device)
        cu_seqlens = torch.repeat_interleave(grid_thw_[:, 1] * grid_thw_[:, 2],
                                             grid_thw_[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        x = x.unsqueeze(1)

        # pre-compute seqlens for attn mask to reduce cuMemcpy operations
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
                seqlens=seqlens,
            )

        # adapter
        x = self.merger(x)

        return x

patch("vllm.model_executor.models.qwen2_vl.Qwen2VisionTransformer", Qwen2VisionTransformerModify).start()