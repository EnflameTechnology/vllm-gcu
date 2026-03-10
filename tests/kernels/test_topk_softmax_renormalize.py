import pytest
import torch
import torch_gcu
from vllm_gcu.kernels import _custom_ops as ops


def ref_topk_softmax_renormalize(gating_output, topk):
    """Reference implementation using PyTorch native ops."""
    softmax_output = torch.softmax(gating_output, dim=1)
    topk_weights, topk_indices = torch.topk(softmax_output, k=topk, dim=1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_tokens, topk, num_experts",
    [
        (5, 4, 10),
        (3, 15, 64),
        (8192, 8, 128),
        (32768, 8, 128),
    ])

def test_topk_softmax_renormalize(dtype, num_tokens, topk, num_experts):
    torch.random.manual_seed(42)

    # Prepare tensors
    topk_weights = torch.zeros(num_tokens, topk, dtype=dtype).gcu()
    topk_indices = torch.zeros_like(topk_weights, dtype=torch.int).gcu()
    token_expert_indices = torch.zeros_like(topk_weights, dtype=torch.int).gcu()
    gating_output = torch.randn(num_tokens, num_experts, dtype=dtype).gcu()
    renormalize = True

    # Run kernel
    torch.ops._moe_C.topk_softmax_renormalize(
        topk_weights, topk_indices, token_expert_indices,
        gating_output, renormalize
    )

    # Compare with reference
    topk_weight_ref, topk_indices_ref = ref_topk_softmax_renormalize(gating_output, topk)
    assert torch.allclose(topk_weight_ref, topk_weights, atol=1e-2, rtol=1e-2)
    # 1% token mismatch is allowed
    exact_match_ratio = (topk_indices_ref == topk_indices).all(dim=1).float().mean()
    assert exact_match_ratio > 0.99, f"Match ratio lower than 99%: {exact_match_ratio}"