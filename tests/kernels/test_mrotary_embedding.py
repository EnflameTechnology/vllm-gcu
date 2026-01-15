import pytest

import torch
import torch_gcu

from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.platforms import current_platform
from vllm_gcu.kernels import _custom_ops as ops


def create_mrope_embedding(head_size, rotary_dim, is_neox, mrope_section, dtype, mrope_interleaved=False):
    """Create a MRotaryEmbedding instance for testing."""
    return MRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=8192,
        base=10000,
        is_neox_style=is_neox,
        dtype=dtype,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_tokens, num_heads, head_size, is_neox",
    [
        (2048, 32,  128, True),
        (1,    32,  128, True),
    ]
)
@pytest.mark.parametrize("mrope_section", [[24, 20, 20]])
def test_interleaved_mrope(dtype, num_tokens, num_heads, head_size, is_neox, mrope_section):
    if current_platform.get_device_capability().to_int() > 130:
        pytest.skip(f"mrope_interleaved is not supported on Libra")
    
    torch.random.manual_seed(42)
    
    rotary_dim = head_size
    
    # Prepare tensors
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype).gcu()
    key = torch.randn(num_tokens, num_heads * head_size, dtype=dtype).gcu()
    positions = torch.stack([
        torch.arange(num_tokens, dtype=torch.long),  # T dimension
        torch.arange(num_tokens, dtype=torch.long),  # H dimension  
        torch.arange(num_tokens, dtype=torch.long),  # W dimension
    ]).gcu()
    
    # Create MRotaryEmbedding instance with interleaved mode
    mrope = create_mrope_embedding(head_size, rotary_dim, is_neox, mrope_section, dtype, mrope_interleaved=True)
    
    # Get cos_sin from cache (cache is on CPU, so we need to use CPU positions)
    cos_sin = mrope.cos_sin_cache[positions.cpu()].gcu()
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    # Run kernel
    query_kernel = query.clone()
    key_kernel = key.clone()
    rotary_dim_kernel = cos.shape[1]
    positions_kernel = torch.arange(rotary_dim_kernel, device=query.device, dtype=torch.long)
    ops.mrotary_embedding(
        positions_kernel,
        query_kernel,
        key_kernel,
        head_size,
        cos_sin,
        is_neox,
        mrope_section,
    )
    
    # Run reference implementation using MRotaryEmbedding.forward_native
    query_ref = query.clone().cpu()
    key_ref = key.clone().cpu()
    positions_ref = positions.cpu()
    query_ref_result, key_ref_result = mrope.forward_native(
        positions_ref,
        query_ref,
        key_ref,
    )
    
    # Compare results
    assert torch.allclose(
        query_ref_result, query_kernel.cpu(), atol=1e-2, rtol=1e-2
    ), "Query mismatch between kernel and reference"
    
    assert torch.allclose(
        key_ref_result, key_kernel.cpu(), atol=1e-2, rtol=1e-2
    ), "Key mismatch between kernel and reference"

