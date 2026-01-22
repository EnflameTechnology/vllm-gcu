import torch
import torch_gcu
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm_gcu.kernels.native_op.utils import register_native

def as_float32_tensor(x: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device="gcu")

@register_native("_C", "dynamic_per_token_scaled_fp8_quant")
def _ref_dynamic_per_token_scaled_fp8_quant(
    output: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, scale_ub: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    # output, input, scale, scale_ub = None

    qtype_traits_min, qtype_traits_max = get_fp8_min_max()
    qtype_max = as_float32_tensor(qtype_traits_max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub is not None:
        x_token_max = x_token_max.clamp(max=scale_ub)
    scales = (x_token_max / qtype_max)[:, None]

    # Quant
    min_scaling_factor = s_1 / (qtype_max * s_512)
    scales = scales.clamp(min=min_scaling_factor)
    torch_out = as_float32_tensor(x) / scales
    torch_out = torch_out.clamp(qtype_traits_min, qtype_traits_max).to(torch.float8_e4m3fn)

    output.copy_(torch_out, True)
    scale.copy_(scales, True)
