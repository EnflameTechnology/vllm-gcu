import pytest
import torch
import torch_gcu
import vllm_gcu._C
from typing import Optional, Dict, Any
import torch.nn.functional as F

class SiluAndMul(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

def torch_experts(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Reference implementation for FP8 per-tensor quantized MoE.
    Based on the original vllm torch_experts implementation, simplified for per-tensor FP8.

    Args:
        a: Input hidden states, already FP8 quantized [M, K]
        w1: First weight matrix, FP8 quantized [E, N, K] where N = 2*intermediate_size
        w2: Second weight matrix, FP8 quantized [E, N', K'] where N' = hidden_size, K' = intermediate_size
        topk_weight: Router weights [M, topk]
        topk_ids: Selected expert IDs [M, topk]
        w1_scale: Per-expert scale for w1 [E]
        w2_scale: Per-expert scale for w2 [E]
        a1_scale: Scale for dequantizing input a
        a2_scale: Scale for dequantizing intermediate activation

    Returns:
        Output hidden states [M, hidden_size]
    """
    M, K = a.shape
    topk = topk_ids.shape[1]

    # Expand input for each expert selection: [M, K] -> [M*topk, K]
    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    out = torch.zeros(M * topk, w2.shape[1], dtype=torch.bfloat16, device=a.device)
    num_experts = w1.shape[0]
    topk_ids = topk_ids.view(-1)
    f32 = torch.float32

    for i in range(num_experts):
        mask = topk_ids == i
        if mask.sum():
            assert (
                a1_scale is not None
                and w1_scale is not None
                and w2_scale is not None
            )

            scales = a1_scale if a1_scale.numel() == 1 else a1_scale[mask]
            tmp1 = a[mask].to(f32) * scales
            w1_dq = (w1[i].to(f32) * w1_scale[i]).transpose(0, 1)
            tmp1 = (tmp1 @ w1_dq).to(out.dtype)

            tmp2 = SiluAndMul()(tmp1).to(out.dtype)

            assert a2_scale is not None
            tmp2_fp8 = (tmp2.to(f32) / a2_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            tmp2_dq = tmp2_fp8.to(f32) * a2_scale
            w2_dq = (w2[i].to(f32) * w2_scale[i]).transpose(0, 1)
            out[mask] = (tmp2_dq @ w2_dq).to(out.dtype)

    return (
        (out.view(M, -1, w2.shape[1]).to(f32) * topk_weight.view(M, -1, 1))
        .sum(dim=1)
        .to(out.dtype)
    )

def create_moe_test_tensors(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    seed: int = 42,
    device: str = 'gcu',
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    fp8_max = 448.0

    # ==================== Original BF16 Tensors ====================
    # Input hidden states: [M, hidden_size]
    hidden_states_bf16 = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.1

    # Expert weights (in BF16 for reference)
    # w1/w13: gate+up projection [E, 2*intermediate_size, hidden_size]
    w1_bf16 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, 
        dtype=torch.bfloat16, device=device
    ) * 0.01

    # w2: down projection [E, hidden_size, intermediate_size]
    w2_bf16 = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device
    ) * 0.01

    # Weight Quantization (per-expert)
    # w1_scale: [E] - one scale per expert
    w1_amax = w1_bf16.abs().amax(dim=(1, 2)) 
    w1_scale = (w1_amax / fp8_max).to(torch.float32).clamp(min=1e-12)
    w1_fp8 = (w1_bf16 / w1_scale.view(-1, 1, 1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    # w2_scale: [E] - one scale per expert
    w2_amax = w2_bf16.abs().amax(dim=(1, 2))
    w2_scale = (w2_amax / fp8_max).to(torch.float32).clamp(min=1e-12)
    w2_fp8 = (w2_bf16 / w2_scale.view(-1, 1, 1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    # Activation Quantization (per-tensor)
    # a1: input activation scale
    a1_amax = hidden_states_bf16.abs().max()
    a1_scale = (a1_amax / fp8_max).to(torch.float32).clamp(min=1e-12)
    a1_scale_rec = (1.0 / a1_scale).to(torch.float32)

    # Quantize input hidden states
    hidden_states_fp8 = (hidden_states_bf16 / a1_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    # a2: intermediate activation scale, here estimated to be 0.1
    a2_scale = torch.tensor(0.0341, dtype=torch.float32, device=device)
    a2_scale_rec = (1.0 / a2_scale).to(torch.float32)

    # Routing
    # topk_weights: [M, topk] - router weights (normalized)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # topk_ids: [M, topk] - selected expert indices
    # Ensure each token selects `topk` different experts
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    for t in range(num_tokens):
        perm = torch.randperm(num_experts, device=device)[:topk]
        topk_ids[t] = perm.to(torch.int32)

    return {
        'hidden_states_bf16': hidden_states_bf16,
        'w1_bf16': w1_bf16,
        'w2_bf16': w2_bf16,
        'hidden_states_fp8': hidden_states_fp8,
        'w1_fp8': w1_fp8,
        'w2_fp8': w2_fp8,
        'w1_scale': w1_scale,
        'w2_scale': w2_scale,
        'a1_scale': a1_scale,
        'a1_scale_rec': a1_scale_rec,
        'a2_scale': a2_scale,
        'a2_scale_rec': a2_scale_rec,
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'num_tokens': num_tokens,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'num_experts': num_experts,
        'topk': topk,
    }

def ref_fused_moe(
    tensors: Dict[str, Any],
) -> torch.Tensor:
    return torch_experts(
        a=tensors['hidden_states_fp8'],
        w1=tensors['w1_fp8'],
        w2=tensors['w2_fp8'],
        topk_weight=tensors['topk_weights'],
        topk_ids=tensors['topk_ids'],
        w1_scale=tensors['w1_scale'],
        w2_scale=tensors['w2_scale'],
        a1_scale=tensors['a1_scale'],
        a2_scale=tensors['a2_scale'],
    )

def run_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple:
    assert topk_ids.dim() == 2

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


def custom_fused_moe(
    tensors: Dict[str, Any],
    block_size: int = 64,
) -> torch.Tensor:
    hidden_states_fp8 = tensors['hidden_states_fp8']
    w1_fp8 = tensors['w1_fp8']
    w2_fp8 = tensors['w2_fp8']
    topk_weights = tensors['topk_weights']
    topk_ids = tensors['topk_ids']
    w1_scale = tensors['w1_scale']
    w2_scale = tensors['w2_scale']
    a1_scale = tensors['a1_scale']
    a2_scale = tensors['a2_scale']
    a2_scale_rec = tensors['a2_scale_rec']
    num_tokens = tensors['num_tokens']
    hidden_size = tensors['hidden_size']
    intermediate_size = tensors['intermediate_size']
    num_experts = tensors['num_experts']
    topk = tensors['topk']

    device = hidden_states_fp8.device

    sorted_token_ids, expert_ids, num_tokens_post_padded = run_moe_align_block_size(
        topk_ids, num_experts, block_size
    )

    # First MoE
    # Input:  hidden_states_fp8 [M, hidden_size]
    # Weight: w1_fp8 [E, 2*intermediate_size, hidden_size]
    # Output: intermediate_cache1 [M, topk, 2*intermediate_size] in bf16
    intermediate_cache1 = torch.empty(
        (num_tokens, topk, 2 * intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )

    torch.ops._C.fused_moe_quant_kernel_ex(
        intermediate_cache1,        # C: output [M, topk, 2*inter]
        hidden_states_fp8,          # A: input [M, hidden_size]
        w1_fp8,                     # B: weight [E, 2*inter, hidden_size]
        a1_scale,                   # A_scale: activation scale (original, for dequant)
        w1_scale,                   # B_scale: weight scale [E]
        None,                       # B_zp: no zero point
        None,                       # bias: no bias
        topk_weights,               # topk_weights: [M, topk]
        topk_ids,                   # topk_ids: [M, topk]
        sorted_token_ids,           # sorted_token_ids from moe_align
        expert_ids,                 # expert_ids from moe_align
        num_tokens_post_padded,     # num_tokens_post_pad
        None,                       # real_token_num: None for dynamic
        False,                      # mul_routed_weight: False for first matmul
        topk,                       # top_k: number of experts per token
        block_size,                 # block_size: 64
        -1,                         # group_k: -1 for per-tensor
        -1,                         # group_n: -1 for per-tensor
    )

    # SiluAndMul + FP8 Quantization
    intermediate_cache2 = torch.empty(
        (num_tokens * topk, intermediate_size),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    expert_num_tokens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    torch.ops._C.silu_mul_static_fp8_quant(
        intermediate_cache2.view(num_tokens, topk, intermediate_size),
        intermediate_cache1,
        a2_scale_rec,
        expert_num_tokens,
    )

    # Second MoE
    # Input:  intermediate_cache2 [M*topk, intermediate_size] in fp8
    # Weight: w2_fp8 [E, hidden_size, intermediate_size]
    # Output: intermediate_cache3 [M, topk, hidden_size] in bf16
    intermediate_cache3 = torch.empty(
        (num_tokens, topk, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )

    torch.ops._C.fused_moe_quant_kernel_ex(
        intermediate_cache3,        # C: output [M, topk, hidden_size]
        intermediate_cache2,        # A: input [M*topk, inter]
        w2_fp8,                     # B: weight [E, hidden_size, inter]
        a2_scale,                   # A_scale: activation scale (original, for dequant)
        w2_scale,                   # B_scale: weight scale [E]
        None,                       # B_zp: no zero point
        None,                       # bias: no bias
        topk_weights,               # topk_weights: [M, topk]
        topk_ids,                   # topk_ids: [M, topk]
        sorted_token_ids,           # sorted_token_ids from moe_align
        expert_ids,                 # expert_ids from moe_align
        num_tokens_post_padded,     # num_tokens_post_pad
        None,                       # real_token_num: None for dynamic
        True,                       # mul_routed_weight: True for second matmul
        1,                          # top_k: 1 for second matmul (special convention)
        block_size,                 # block_size: 64
        -1,                         # group_k: -1 for per-tensor
        -1,                         # group_n: -1 for per-tensor
    )


    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )

    torch.ops._moe_C.moe_sum_pad(
        output,
        intermediate_cache3,
        expert_num_tokens,
        1,
        False,
    )

    return output

def test_fused_moe_full(
    num_tokens: int = 128,
    num_experts: int = 64,
    topk: int = 8,
    hidden_size: int = 4096,
    intermediate_size: int = 3072,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    print(f"\n{'='*70}")
    print(f"Test: num_tokens={num_tokens}, experts={num_experts}, topk={topk}")
    print(f"      hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    print(f"{'='*70}")

    tensors = create_moe_test_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    )

    print(f"\nTest tensors created:")
    print(f"  hidden_states_fp8: {tensors['hidden_states_fp8'].shape}")
    print(f"  w1_fp8: {tensors['w1_fp8'].shape}")
    print(f"  w2_fp8: {tensors['w2_fp8'].shape}")
    print(f"  topk_ids: {tensors['topk_ids'].shape}")
    print(f"  a1_scale: {tensors['a1_scale']}")
    print(f"  a2_scale: {tensors['a2_scale']}")

    ref_out = ref_fused_moe(tensors)

    assert not torch.isnan(ref_out).any(), "Reference output contains NaN"
    assert not torch.isinf(ref_out).any(), "Reference output contains Inf"

    custom_out = custom_fused_moe(tensors, block_size=64)

    assert not torch.isnan(custom_out).any(), "Custom output contains NaN"
    assert not torch.isinf(custom_out).any(), "Custom output contains Inf"


    diff = (custom_out - ref_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    ref_abs = ref_out.abs()
    rel_diff = diff / (ref_abs + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"  Absolute difference:")
    print(f"    Max:  {max_diff:.6f}")
    print(f"    Mean: {mean_diff:.6f}")
    print(f"  Relative difference:")
    print(f"    Max:  {max_rel_diff:.6f}")
    print(f"    Mean: {mean_rel_diff:.6f}")

    is_close = torch.allclose(custom_out, ref_out, rtol=rtol, atol=atol)

    if is_close:
        print(f"\n[PASS] Outputs match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"\n[FAIL] Outputs do NOT match within tolerance")
        print(f"  Expected rtol={rtol}, atol={atol}")
        print(f"  Got max_diff={max_diff:.6f}, max_rel_diff={max_rel_diff:.6f}")

    assert is_close, f"Output mismatch: max_diff={max_diff:.6f}, max_rel_diff={max_rel_diff:.6f}"

    return ref_out, custom_out


@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("hidden_size, intermediate_size", [(4096, 3072)])
def test_fused_moe_fp8_per_tensor(num_tokens, num_experts, topk, hidden_size, intermediate_size):
    test_fused_moe_full(num_tokens, num_experts, topk, hidden_size, intermediate_size)


if __name__ == "__main__":
    test_configs = [
        # (num_tokens, num_experts, topk, hidden_size, intermediate_size)
        (1, 64, 8, 4096, 3072),
        (32, 64, 8, 4096, 3072),
        (128, 64, 8, 4096, 3072),
        (8192, 64, 8, 4096, 3072),
        (32768, 64, 8, 4096, 3072),
    ]

    passed = 0
    failed = 0

    for num_tokens, num_experts, topk, hidden_size, intermediate_size in test_configs:
        try:
            test_fused_moe_full(
                num_tokens=num_tokens,
                num_experts=num_experts,
                topk=topk,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rtol=1e-1,
                atol=1e-4,
            )
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] tokens={num_tokens}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)