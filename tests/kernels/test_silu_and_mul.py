import pytest
import torch_gcu
import torch
from vllm_gcu.kernels import _custom_ops as ops
from vllm_gcu.kernels.native_op.torch_native_op import register_native_overrides

def _run_kernel(input_tensor: torch.Tensor) -> torch.Tensor:
    d = input_tensor.shape[-1] // 2
    out = torch.empty(input_tensor.shape[:-1] + (d,),
                      dtype=input_tensor.dtype,
                      device=input_tensor.device)
    torch.ops._C.silu_and_mul(out, input_tensor)
    return out

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@torch.inference_mode()
def test_silu_and_mul_compare_via_override(dtype):
    torch.random.manual_seed(42)
    shape = (4, 8, 16)
    x = torch.randn(shape, dtype=dtype).gcu()

    # kernel 输出
    out_kernel = _run_kernel(x)

    # 覆盖实现为 python 参考版本
    register_native_overrides({"fallback_ops": ["silu_and_mul"]})
    out_override = _run_kernel(x)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    torch.testing.assert_close(out_kernel, out_override, rtol=rtol, atol=atol)