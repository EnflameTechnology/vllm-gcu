import pytest
import torch
import torch_gcu
import vllm_gcu._C


def ref_linear_quant_fp8_per_tensor(
    x_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Reference implementation for FP8 per-tensor quantized linear.

    FP8 per-tensor quantization:
    - x_fp8 = quantize(x_original / x_scale)
    - weight_fp8 = quantize(weight_original / w_scale)

    Dequantized matmul:
    - output = (x_fp8 * x_scale) @ (weight_fp8 * w_scale).T + bias

    Args:
        x_fp8: Input activation in FP8, shape (M, K)
        weight_fp8: Weight in FP8, shape (N, K)
        x_scale: Per-tensor scale for input, scalar or shape (1,)
        w_scale: Per-tensor scale for weight, scalar or shape (1,)
        bias: Optional bias, shape (N,)
        out_dtype: Output dtype

    Returns:
        output: shape (M, N)
    """
    # Dequantize to FP32
    x_fp32 = x_fp8.to(torch.float32) * x_scale.to(torch.float32)
    w_fp32 = weight_fp8.to(torch.float32) * w_scale.to(torch.float32)

    # Linear: (M, K) @ (N, K).T -> (M, N)
    output = x_fp32 @ w_fp32.T

    if bias is not None:
        output = output + bias.to(torch.float32)

    return output.to(out_dtype)


def create_fp8_test_tensors(M, K, N, x_scale_dtype=torch.bfloat16, w_scale_dtype=torch.float32):
    """
    Create test tensors for FP8 per-tensor quantization test.

    Simulates the quantization process:
    1. Create original BF16 tensors
    2. Compute scales based on tensor range
    3. Quantize to FP8

    Returns quantized tensors and scales.
    """
    torch.manual_seed(42)

    # Create bf16 weight and activation tensors
    x_original = torch.randn(M, K, dtype=torch.bfloat16)
    weight_original = torch.randn(N, K, dtype=torch.bfloat16)

    # Compute per-tensor scales
    fp8_max = 448.0
    x_scale = (x_original.abs().max() / fp8_max).to(x_scale_dtype)
    w_scale = (weight_original.abs().max() / fp8_max).to(w_scale_dtype)

    # Ensure scales are not zero
    x_scale = torch.clamp(x_scale, min=1e-6)
    w_scale = torch.clamp(w_scale, min=1e-6)

    # Quantize to FP8
    x_fp8 = (x_original / x_scale.to(torch.float32)).to(torch.float8_e4m3fn)
    weight_fp8 = (weight_original / w_scale.to(torch.float32)).to(torch.float8_e4m3fn)

    return x_fp8, weight_fp8, x_scale, w_scale


@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 4096, 4096),
        (32, 4096, 4096),
        (8192, 4096, 4096),
        (32768, 4096, 4096),
        (1, 4096, 1024),
        (32, 4096, 1024),
        (8192, 4096, 1024),
        (32768, 4096, 1024),
        (1, 4096, 6144),
        (32, 4096, 6144),
        (8192, 4096, 6144),
        (32768, 4096, 6144),
        (1, 4096, 3072),
        (32, 4096, 3072),
        (8192, 4096, 3072),
        (32768, 4096, 3072),
        (1, 3072, 4096),
        (32, 3072, 4096),
        (8192, 3072, 4096),
        (32768, 3072, 4096),
    ])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_quant_per_tensor_fp8(out_dtype, M, K, N, with_bias):
    # Create test tensors
    x_fp8, weight_fp8, x_scale, w_scale = create_fp8_test_tensors(
        M, K, N, 
        x_scale_dtype=torch.bfloat16,
        w_scale_dtype=torch.float32
    )

    # bias
    bias = None
    if with_bias:
        bias = torch.randn(N, dtype=out_dtype)

    # Move tensors to GCU
    x_fp8_gcu = x_fp8.gcu()
    weight_fp8_gcu = weight_fp8.gcu()
    x_scale_gcu = x_scale.gcu()
    w_scale_gcu = w_scale.gcu()
    bias_gcu = bias.gcu() if bias is not None else None

    # Compute reference output
    ref_output = ref_linear_quant_fp8_per_tensor(
        x_fp8_gcu, weight_fp8_gcu, x_scale_gcu, w_scale_gcu, bias_gcu, out_dtype
    )

    output_gcu = torch.empty(M, N, dtype=out_dtype, device='gcu')

    # Use cutlass_scaled_mm interface
    # This should dispatch to topsextsLinearQuant for FP8 weight
    # Signature: cutlass_scaled_mm(out, x, weight, x_scale, w_scale, bias)
    torch.ops._C.cutlass_scaled_mm(
        output_gcu,
        x_fp8_gcu,
        weight_fp8_gcu,
        x_scale_gcu,
        w_scale_gcu,
        bias_gcu
    )

    # Compare results
    output_cpu = output_gcu.cpu()
    ref_output_cpu = ref_output.cpu()

    # FP8 has limited precision, so we expect some error
    atol = 1e-3  # Absolute tolerance
    rtol = 1e-2  # Relative tolerance

    # Check if outputs are close
    if not torch.allclose(output_cpu, ref_output_cpu, atol=atol, rtol=rtol):
        # Print debug info if test fails
        max_diff = (output_cpu - ref_output_cpu).abs().max()
        mean_diff = (output_cpu - ref_output_cpu).abs().mean()
        print(f"\nTest failed for M={M}, K={K}, N={N}, with_bias={with_bias}")
        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"Output sample:\n{output_cpu[0, :5]}")
        print(f"Ref sample:\n{ref_output_cpu[0, :5]}")

    assert torch.allclose(output_cpu, ref_output_cpu, atol=atol, rtol=rtol), \
        f"Output mismatch! Max diff: {(output_cpu - ref_output_cpu).abs().max():.6f}"


if __name__ == "__main__":
    test_shapes = [
        (1, 4096, 4096),
        (32, 4096, 4096),
        (8192, 4096, 4096),
        (32768, 4096, 4096),
        (1, 4096, 1024),
        (32, 4096, 1024),
        (8192, 4096, 1024),
        (32768, 4096, 1024),
        (1, 4096, 6144),
        (32, 4096, 6144),
        (8192, 4096, 6144),
        (32768, 4096, 6144),
        (1, 4096, 3072),
        (32, 4096, 3072),
        (8192, 4096, 3072),
        (32768, 4096, 3072),
        (1, 3072, 4096),
        (32, 3072, 4096),
        (8192, 3072, 4096),
        (32768, 3072, 4096),
    ]
    bias_options = [False, True]
    out_dtype = torch.bfloat16

    total = len(test_shapes) * len(bias_options)
    passed = 0
    failed = 0

    print(f"Running {total} test cases...")
    print("=" * 60)

    for M, K, N in test_shapes:
        for with_bias in bias_options:
            test_name = f"M={M}, K={K}, N={N}, bias={with_bias}"
            try:
                test_linear_quant_per_tensor_fp8(out_dtype, M, K, N, with_bias)
                print(f"[PASS] {test_name}")
                passed += 1
            except Exception as e:
                print(f"[FAIL] {test_name}")
                print(f"       Error: {e}")
                failed += 1

    print("=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
