import torch
from vllm_gcu.kernels.native_op.utils import register_native
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max

def as_float32_tensor(x: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device="gcu")

@register_native("_C", "dynamic_scaled_fp8_quant")
def _ref_dynamic_scaled_fp8_quant(output: torch.Tensor, input: torch.Tensor, scale: torch.Tensor):
    fp8_traits_min, fp8_traits_max = get_fp8_min_max()
    fp8_max = as_float32_tensor(fp8_traits_max)
    one = as_float32_tensor(1.0)

    x_max = as_float32_tensor(input.abs().max())
    ref_scale = x_max / fp8_max
    ref_iscale = one / ref_scale
    ref_out = (
        (as_float32_tensor(input) * ref_iscale)
        .clamp(fp8_traits_min, fp8_traits_max)
        .to(torch.float8_e4m3fn)
    )
    output.copy_(ref_out, True)
    scale.copy_(ref_scale.view(1), True)