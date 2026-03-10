import torch
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_moe_C", "topk_softmax_renormalize")
def _ref_topk_softmax_renormalize(
    topk_weights_: torch.Tensor,
    topk_indices_: torch.Tensor,
    token_expert_indices_: torch.Tensor,
    gating_output_: torch.Tensor,
    renormalize_: bool = True,
):
    """Reference implementation using PyTorch native ops."""
    softmax_output = torch.softmax(gating_output_, dim=1)
    topk_weights, topk_indices = torch.topk(softmax_output, k=topk_indices_.size(1), dim=1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights_.copy_(topk_weights)
    topk_indices_.copy_(topk_indices)
