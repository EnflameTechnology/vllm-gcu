import torch
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C", "silu_and_mul")
def _ref_silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    d = input.shape[-1] // 2
    left = input[..., :d]
    right = input[..., d:]
    out.copy_(left * torch.sigmoid(left) * right)