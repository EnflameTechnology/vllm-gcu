import torch

from vllm_gcu.kernels.native_op.utils import register_native

@register_native("_C", "per_token_group_fp8_quant_packed")
def _ref_per_token_group_fp8_quant_packed(
    x_contiguous: torch.Tensor,
    x_q_local: torch.Tensor,
    x_s_packed: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float
):
    k = x_contiguous.shape[-1]

    mn = x_contiguous.numel() // k
    group_per_row = k // group_size
    # kernel use for groups_per_block and num_blocks
    # no need in native impl
    # num_groups = mn * group_per_row

    k_num_packed_sfk = (group_per_row + 3 ) // 4

    # tma_aligned_mn for stride
    # out_idx = sf_k_pack_idx * tma_aligned_mn + mn_idx
    # no need in naive impl
    # tma_aligned_mn = ((mn+3) //4 ) * 4

    x_grouped = x_contiguous.reshape(mn, group_per_row, group_size).float()
    group_absmax = x_grouped.abs().amax(dim=-1).clamp(min=eps)

    # calc scale and turn scale to ue8m0
    y_s = group_absmax / fp8_max
    y_s = torch.exp2(torch.ceil(torch.log2(y_s.clamp(min=1e-10))))

    #  quantize
    y_s_expanded = y_s.unsqueeze(-1)
    q = (x_grouped / y_s_expanded).clamp(fp8_min, fp8_max)
    x_q_local.copy_(q.reshape(mn, k).to(torch.float8_e4m3fn))

    y_s_bits = y_s.contiguous().view(torch.int32)
    exponents = (y_s_bits >> 23) & 0xFF

    padded = torch.zeros(mn, k_num_packed_sfk * 4, dtype=torch.int32, device=x_contiguous.device)
    padded[:, :group_per_row] = exponents

    padded = padded.reshape(mn, k_num_packed_sfk, 4)  # [mn, k_num_packed_sfk, 4]

    # pack: byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    packed = (padded[:, :, 0]
            | (padded[:, :, 1] << 8)
            | (padded[:, :, 2] << 16)
            | (padded[:, :, 3] << 24))  # [mn, k_num_packed_sfk]

    x_s_packed.copy_(packed)