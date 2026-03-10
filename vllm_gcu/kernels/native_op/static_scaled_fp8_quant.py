import torch
from vllm.platforms import current_platform
from vllm_gcu.kernels.native_op.utils import register_native

def get_fp8_min_max() -> tuple[float, float]:
    """Get the min and max values for FP8 quantization."""
    # Using the default value (240.0) from pytorch will cause accuracy
    # issue on dynamic quantization models on ROCm. Here, use 224.0 for fnuz
    # on ROCm platforms that use the torch.float8_e4m3fnuz dtype.
    if current_platform.is_fp8_fnuz():
        return -224.0, 224.0
    finfo = torch.finfo(current_platform.fp8_dtype())
    return finfo.min, finfo.max

def _normalize_quant_group_shape(x: torch.Tensor, group_shape: GroupShape):
    # -1 means full extent
    return (
        group_shape[0] if group_shape[0] > 0 else x.shape[-2],
        group_shape[1] if group_shape[1] > 0 else x.shape[-1],
    )

@register_native("_C", "static_scaled_fp8_quant")
def _ref_static_scaled_fp8_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_shape: GroupShape,
):
    group_shape = _normalize_quant_group_shape(input, group_shape)

    assert input.ndim == 2
    assert input.shape[0] % group_shape[0] == 0 and input.shape[1] % group_shape[1] == 0
    blk_m, blk_n = input.shape[0] // group_shape[0], input.shape[1] // group_shape[1]
    input_blkd = input.reshape(blk_m, group_shape[0], blk_n, group_shape[1])
    input_blkd_permd = input_blkd.permute(0, 2, 1, 3)
    input_blkd_permd = input_blkd_permd.flatten(start_dim=2)

    min_val, max_val = input_blkd_permd.aminmax(dim=-1)
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    _, fp8_max = get_fp8_min_max()
    scale = fp8_max / amax

    x_scl_sat = (
        (input_blkd_permd * scale.unsqueeze(-1))
        .clamp(min=finfo.min, max=finfo.max)
        .reshape(blk_m, blk_n, group_shape[0], group_shape[1])
        .permute(0, 2, 1, 3)
        .reshape(input.shape)
    )

    output.copy_(x_scl_sat, True)
    scale.copy_(scale.float().reciprocal(), True)
