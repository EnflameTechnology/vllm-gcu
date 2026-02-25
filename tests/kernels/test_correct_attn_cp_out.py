import pytest
import torch
import triton
import triton.language as tl

import triton_gcu.triton
import torch_gcu  # noqa: F401
import torch_gcu.transfer_to_gcu  # noqa: F401
from vllm.attention.ops.common import correct_attn_out


def ref_correct_attn_cp_out(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    lse_idx: int,
    is_base_e: bool,
):
    """
    PyTorch implementation of `_correct_attn_cp_out_kernel`.

    Args:
        outputs (torch.Tensor): Input tensor of shape [B, H, D].
        lses (torch.Tensor): Input tensor of shape [N, B, H].
        lse_idx (int): Index for accessing specific LSE values.
        is_base_e (bool): Whether to use base-e (exp/log) or base-2 (exp2/log2).

    Returns:
        new_output (torch.Tensor): Corrected output tensor of shape [B, H, D].
        vlse (torch.Tensor): Final LSE tensor of shape [B, H].
    """
    # Dimensions
    B, H, D = outputs.shape
    N = lses.shape[0]

    # Step 1: Calculate final LSE
    lse = lses.view(N, B, H)  # Shape [N, B, H]
    lse = torch.where(
        torch.isnan(lse) | torch.isinf(lse),
        torch.tensor(-float("inf"), device=lse.device, dtype=lse.dtype),
        lse,
    )

    # Compute the max value along N-axis for numerical stability
    lse_max = torch.max(lse, dim=0).values  # Shape [B, H]
    lse_max = torch.where(
        lse_max == -float("inf"),
        torch.tensor(0.0, device=lse.device, dtype=lse.dtype),
        lse_max,
    )

    # Subtract the max value for stability
    lse -= lse_max  # Broadcasting over [N, B, H]

    if is_base_e:
        lse_exp = torch.exp(lse)  # Shape [N, B, H]
        lse_acc = torch.sum(lse_exp, dim=0)  # Shape [B, H]
        lse_final = torch.log(lse_acc)  # Shape [B, H]
    else:
        lse_exp = torch.exp2(lse)  # Shape [N, B, H]
        lse_acc = torch.sum(lse_exp, dim=0)  # Shape [B, H]
        lse_final = torch.log2(lse_acc)  # Shape [B, H]

    # Add back the max value
    vlse = lse_final + lse_max  # Shape [B, H]

    # Step 2: Correct the output tensor
    lse_tmp = lses[lse_idx]  # Shape [B, H]
    lse_diff = lse_tmp - vlse  # Shape [B, H]

    lse_diff = torch.where(
        torch.isnan(lse_diff) | torch.isinf(lse_diff),
        torch.tensor(-float("inf"), device=lse.device, dtype=lse.dtype),
        lse_diff,
    )

    # Compute the factor
    factor = (
        torch.exp(lse_diff) if is_base_e else torch.exp2(lse_diff)
    )  # Shape [B, H, 1]

    # Reshape `factor` for broadcasting and apply it to `outputs`
    factor = factor.unsqueeze(-1)  # Shape [B, H, 1]
    new_output = outputs * factor  # Shape [B, H, D]

    return new_output, vlse


@pytest.mark.parametrize("b", [1, 4, 32])
@pytest.mark.parametrize("h", [32, 128])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("cp_size", [4])
@pytest.mark.parametrize("cp_rank", [0, 1, 2, 3])
def test_correct_attn_cp_out(b, h, d, cp_size, cp_rank):
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    out = torch.randn((b, h, d))
    lses = torch.randn((cp_size, b, h))

    out_test = out.clone()  # inplace
    out_test, lse_test = correct_attn_out(out_test, lses, cp_rank, None)
    out_ref, lse_ref = ref_correct_attn_cp_out(out, lses, cp_rank, True)
    assert torch.allclose(out_test, out_ref, 1e-1, 1e-2)
    assert torch.allclose(lse_test, lse_ref, 1e-1, 1e-2)


if __name__ == "__main__":
    test_correct_attn_cp_out(1, 32, 128, 4, 0)
