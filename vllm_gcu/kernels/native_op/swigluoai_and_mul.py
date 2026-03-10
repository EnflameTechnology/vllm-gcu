import torch
from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C", "swigluoai_and_mul")
def _ref_swigluoai_and_mul(out: torch.Tensor, input: torch.Tensor,  alpha: float = 1.702,  limit: float = 7.0) -> None:
    gate, up = input[..., ::2], input[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    out.copy_(gated_output)