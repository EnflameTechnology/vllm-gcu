import pytest

import torch
import torch_gcu

from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm_gcu.kernels import _custom_ops as ops
from vllm.platforms import current_platform


def create_mrope_embedding(head_size, rotary_dim, is_neox, mrope_section, dtype, mrope_interleaved):
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


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("is_neox", [True])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("mrope_interleaved", [True, False])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize(
    "num_tokens", [1, 313, 6474, 8192]
)
@pytest.mark.parametrize("mrope_section", [[24, 20, 20]])
def test_mrope(dtype, num_tokens: int, num_heads: int, head_size: int, group_size: int, is_neox: bool, mrope_section: list[int], mrope_interleaved: bool):
    
    torch.random.manual_seed(42)
    
    rotary_dim = head_size
    
    # Prepare tensors
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype).gcu()
    key = torch.randn(num_tokens, num_heads * head_size // group_size, dtype=dtype).gcu()

    positions = torch.stack([
        torch.randint(0, num_tokens, (num_tokens,), dtype=torch.int64),  # T dimension
        torch.randint(0, num_tokens, (num_tokens,), dtype=torch.int64),  # H dimension  
        torch.randint(0, num_tokens, (num_tokens,), dtype=torch.int64),  # W dimension
    ]).gcu().contiguous()  # [3, num_tokens]

    # Create MRotaryEmbedding instance with interleaved mode
    mrope = create_mrope_embedding(head_size, rotary_dim, is_neox, mrope_section, dtype, mrope_interleaved)
    mrope.cos_sin_cache = mrope.cos_sin_cache.gcu()
    
    # Run kernel
    query_kernel = query.clone()
    key_kernel = key.clone()
    ops.mrotary_embedding(
        positions,
        query_kernel,
        key_kernel,
        head_size,
        mrope.cos_sin_cache,
        is_neox,
        mrope_section,
        mrope.mrope_interleaved,
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
    if current_platform.get_device_capability().to_int() > 130:
        assert torch.allclose(
            query_ref_result, query_kernel.cpu(), atol=1e-2, rtol=1e-2
        ), "Query mismatch between kernel and reference"
        
        assert torch.allclose(
            key_ref_result, key_kernel.cpu(), atol=1e-2, rtol=1e-2
        ), "Key mismatch between kernel and reference"
    else:
        assert torch.allclose(
            query_ref_result, query_kernel.cpu(), atol=1e-1, rtol=1e-1
        ), "Query mismatch between kernel and reference"
        
        assert torch.allclose(
            key_ref_result, key_kernel.cpu(), atol=1e-1, rtol=1e-1
        ), "Key mismatch between kernel and reference"
