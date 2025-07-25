from typing import Optional, Union
import torch
# import torch_gcu

from vllm import _custom_ops as ops
# from vllm_gcu.kernels import _custom_ops as ops

from vllm.platforms import current_platform


def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='gcu')

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.tensor] = None) \
        -> tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.int8, torch.float8_e4m3fn]
    if scale_ub:
        assert quant_dtype == torch.float8_e4m3fn

    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(quant_dtype)
    qtype_traits_max = qtype_traits.max
    qtype_traits_min = qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # For fp8, in order to match the gcu kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    # Compute scales
    # differ from gpu
    x_token_max = x.abs().max(dim=-1, keepdim=True)[0]
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub:
        x_token_max = x_token_max.clamp(max=scale_ub)
    # differ from gpu
    scales = (x_token_max / qtype_max)

    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        torch_out = as_float32_tensor(x) * iscales
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    else:
        assert quant_dtype == torch.float8_e4m3fn
        min_scaling_factor = s_1 / (qtype_max * s_512)
        #differ from gpu
        # scales = scales.clamp(min=min_scaling_factor)
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)

    return torch_out, scales
