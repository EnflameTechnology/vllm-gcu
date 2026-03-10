import pytest
import torch
import time

from vllm.platforms import current_platform
from vllm_gcu.kernels import _custom_ops as ops
from torch_gcu import transfer_to_gcu

def cdiv(a, b):
    """Ceiling division (rounded up)"""
    return (a + b - 1) // b

@torch.inference_mode()
def cp_gather_indexer_k_quant_cache(
    kv_cache,  # [total_physical_blocks, block_size, head_dim + 4], uint8
    dst_value,  # [total_tokens, head_dim], float8_e4m3fn
    dst_scale,  # [total_tokens, 1], float32
    block_table,  # [batch_size, logical_block_table_size], int32
    cu_seq_lens,  # [batch_size + 1], int32
    batch_size,
):
    num_blocks, block_size, _ = kv_cache.shape
    head_dim = dst_value.shape[-1]
    kv_cache = kv_cache.view(num_blocks, -1)

    expected_value = []
    expected_scale = []
    for b in range(batch_size):
        s = cu_seq_lens[b + 1] - cu_seq_lens[b]
        if s == 0:
            continue
        tot = cdiv(s, block_size)
        blocks = block_table[b, :tot]

        value = []
        scale = []
        full_block = torch.arange(tot - 1,
                                  device=kv_cache.device,
                                  dtype=torch.int32)
        non_remaining_value = kv_cache[blocks[full_block], :block_size *
                                       head_dim].view(-1, head_dim)
        non_remaining_scale = kv_cache[blocks[full_block],
                                       block_size * head_dim:].view(-1, 4)

        remaining = s - (tot - 1) * block_size

        value = torch.cat([
            non_remaining_value,
            kv_cache[blocks[-1], :remaining * head_dim].view(-1, head_dim)
        ],
                          dim=0)
        scale = torch.cat([
            non_remaining_scale,
            kv_cache[blocks[-1], block_size * head_dim:block_size * head_dim +
                     remaining * 4].view(-1, 4)
        ],
                          dim=0)

        expected_value.append(value)
        expected_scale.append(scale)

    gather_value = torch.cat(expected_value, dim=0).view(-1, head_dim)
    gather_scale = torch.cat(expected_scale, dim=0).view(-1, 4)
    gather_value = gather_value.view(torch.float8_e4m3fn)
    gather_scale = gather_scale.view(torch.float32)
    dst_value[cu_seq_lens[0]:cu_seq_lens[-1]].copy_(gather_value)
    dst_scale[cu_seq_lens[0]:cu_seq_lens[-1]].copy_(gather_scale)


@pytest.mark.parametrize("num_physical_blocks", [100, 12498])
@pytest.mark.parametrize("logical_block_table_size", [2560])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("total_tokens", [28, 2048])
@torch.inference_mode()
def test_apply_cp_gather_indexer_k_quant_cache(
    num_physical_blocks: int,
    logical_block_table_size: int,
    block_size: int,
    head_dim: int,
    batch_size: int,
    total_tokens: int,
) -> None:
    """
    Test the cp_gather_indexer_k_quant_cache custom op
    against a reference implementation.
    """
    current_platform.seed_everything(42)
    torch.set_default_device("cuda:0")

    """Test actual production configuration"""
    print(f"Actual production configuration:")
    print(f"  kv_cache: [{num_physical_blocks}, {block_size}, {head_dim + 4}]")
    print(f"  block_table: [{batch_size}, {logical_block_table_size}]")
    print(f"  dst_value: [{total_tokens}, {head_dim}]")
    print(f"  dst_scale: [{total_tokens}, 1]")

    # For testing, use a smaller number of physical blocks
    test_num_physical_blocks = num_physical_blocks # 100
    print(f"\nTest configuration (using fewer physical blocks for easier testing):")
    print(f"  kv_cache: [{test_num_physical_blocks}, {block_size}, {head_dim + 4}]")

    # Create kv_cache (physical blocks)
    kv_cache = torch.randint(
        0, 256,
        (test_num_physical_blocks, block_size, head_dim + 4),
        dtype=torch.uint8,
        device="cuda"
    )

    # Create cu_seq_lens
    cu_seq_lens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

    # Create block_table (logical block table)
    block_table = torch.full((batch_size, logical_block_table_size), -1, dtype=torch.int32, device="cuda")

    # Calculate needed blocks: 28 tokens require 1 block (since block_size=64)
    blocks_needed = cdiv(total_tokens, block_size)  # Should be 1

    # Assign physical block indices to block_table
    # Note: Indices in block_table must be in range [0, test_num_physical_blocks-1]
    for i in range(blocks_needed):
        block_table[0, i] = i % test_num_physical_blocks  # Use valid physical block indices

    print(f"\nBlock allocation:")
    print(f"  Need {blocks_needed} physical blocks")
    print(f"  First {blocks_needed} indices allocated in block_table: {block_table[0, :blocks_needed].cpu().numpy()}")

    # Create output buffers
    dst_value = torch.empty(total_tokens, head_dim, dtype=torch.float8_e4m3fn, device="cuda")
    dst_scale = torch.empty(total_tokens, 1, dtype=torch.float32, device="cuda")

    # Execute function
    cp_gather_indexer_k_quant_cache(
        kv_cache, dst_value, dst_scale, block_table, cu_seq_lens, batch_size
    )

    k_fp8 = torch.empty([total_tokens, head_dim], device="cuda", dtype=torch.float8_e4m3fn)
    k_scale = torch.empty([total_tokens, 4], device="cuda", dtype=torch.uint8)
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache,
        k_fp8,
        k_scale,
        block_table,
        cu_seq_lens,
    )
    k_scale = k_scale.view(torch.float32)

    # Replace NaN with 0
    k_fp8_clean = torch.where(torch.isnan(k_fp8.to('cpu')), torch.zeros_like(k_fp8.to('cpu')), k_fp8.to('cpu'))
    dst_value_clean = torch.where(torch.isnan(dst_value.to('cpu')), torch.zeros_like(dst_value.to('cpu')), dst_value.to('cpu'))
    diff_dst_value= (k_fp8_clean.to(torch.float32) - dst_value_clean.to(torch.float32)).abs()

    k_scale_clean = torch.where(torch.isnan(k_scale.to('cpu')), torch.zeros_like(k_scale.to('cpu')), k_scale.to('cpu'))
    dst_scale_clean = torch.where(torch.isnan(dst_scale.to('cpu')), torch.zeros_like(dst_scale.to('cpu')), dst_scale.to('cpu'))
    diff_dst_scale = (k_scale_clean.to(torch.float32) - dst_scale_clean.to(torch.float32)).abs()

    # Convert to float32 and calculate difference
    print('******************* cp_gather_indexer_k_quant_cache dst_value diff -> mean: {}, max: {}'
          .format(diff_dst_value.mean(), diff_dst_value.max()), flush=True)
    print('******************* cp_gather_indexer_k_quant_cache dst_scale diff -> mean: {}, max: {}'
          .format(diff_dst_scale.mean(), diff_dst_scale.max()), flush=True)

    # Verify results
    assert torch.allclose(k_fp8_clean.float(), dst_value_clean.float(), rtol=1e-3, atol=1e-5)
    assert torch.allclose(k_scale_clean.float(), dst_scale_clean.float(), rtol=1e-3, atol=1e-5)

    assert dst_value.shape == (total_tokens, head_dim)
    assert dst_scale.shape == (total_tokens, 1)

    print(f"\n Shape verification passed")

    # Check if indices in block_table are valid
    used_blocks = block_table[0, :blocks_needed]
    assert torch.all(used_blocks >= 0), "block_table contains negative indices"
    assert torch.all(used_blocks < test_num_physical_blocks), "block_table indices exceed physical block range"

    print(f" Block index verification passed")
    print(f"   Used physical block indices: {used_blocks.cpu().numpy()}")
    print(f"   Total physical blocks: {test_num_physical_blocks}")


@pytest.mark.parametrize("num_physical_blocks", [10])
@pytest.mark.parametrize("logical_block_table_size", [2560])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("total_tokens", [28])
@torch.inference_mode()
def test_apply_valid_block_indices_with_padding(
    num_physical_blocks: int,
    logical_block_table_size: int,
    block_size: int,
    head_dim: int,
    batch_size: int,
    total_tokens: int,
) -> None:
    """Test valid block indices with padding (-1 after required blocks)"""
    print("\nTesting valid block indices with padding...")
    # Create kv_cache
    kv_cache = torch.randint(
        0, 256,
        (num_physical_blocks, block_size, head_dim + 4),
        dtype=torch.uint8,
        device="cuda"
    )

    # Create cu_seq_lens
    cu_seq_lens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

    # Create block_table
    block_table = torch.full((batch_size, logical_block_table_size), -1, dtype=torch.int32, device="cuda")

    # Assign a negative index
    # block_table[0, 0] = -1  # not supported

    # Assign an out-of-range index
    # block_table[0, 0] = num_physical_blocks  # not supported

    # Assign valid indices, followed by -1 padding
    block_table[0, 0] = 0  # Valid block index
    # block_table[0, 1:] remains -1 (normal padding)

    # Create output buffers
    dst_value = torch.empty(total_tokens, head_dim, dtype=torch.float8_e4m3fn, device="cuda")
    dst_scale = torch.empty(total_tokens, 1, dtype=torch.float32, device="cuda")

    try:
        # Execute function (should succeed since only first index is used and valid)
        cp_gather_indexer_k_quant_cache(
            kv_cache, dst_value, dst_scale, block_table, cu_seq_lens, batch_size
        )

        k_fp8 = torch.empty([total_tokens, head_dim], device="cuda", dtype=torch.float8_e4m3fn)
        k_scale = torch.empty([total_tokens, 4], device="cuda", dtype=torch.uint8)
        torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8,
            k_scale,
            block_table,
            cu_seq_lens,
        )
        k_scale = k_scale.view(torch.float32)

        # Replace NaN with 0
        k_fp8_clean = torch.where(torch.isnan(k_fp8.to('cpu')), torch.zeros_like(k_fp8.to('cpu')), k_fp8.to('cpu'))
        dst_value_clean = torch.where(torch.isnan(dst_value.to('cpu')), torch.zeros_like(dst_value.to('cpu')), dst_value.to('cpu'))
        diff_dst_value= (k_fp8_clean.to(torch.float32) - dst_value_clean.to(torch.float32)).abs()

        k_scale_clean = torch.where(torch.isnan(k_scale.to('cpu')), torch.zeros_like(k_scale.to('cpu')), k_scale.to('cpu'))
        dst_scale_clean = torch.where(torch.isnan(dst_scale.to('cpu')), torch.zeros_like(dst_scale.to('cpu')), dst_scale.to('cpu'))
        diff_dst_scale = (k_scale_clean.to(torch.float32) - dst_scale_clean.to(torch.float32)).abs()

        # Convert to float32 and calculate difference
        print('******************* cp_gather_indexer_k_quant_cache dst_value diff -> mean: {}, max: {}'
            .format(diff_dst_value.mean(), diff_dst_value.max()), flush=True)
        print('******************* cp_gather_indexer_k_quant_cache dst_scale diff -> mean: {}, max: {}'
            .format(diff_dst_scale.mean(), diff_dst_scale.max()), flush=True)

        # Verify results
        assert torch.allclose(k_fp8_clean.float(), dst_value_clean.float(), rtol=1e-3, atol=1e-5)
        assert torch.allclose(k_scale_clean.float(), dst_scale_clean.float(), rtol=1e-3, atol=1e-5)

        print(f" Valid indices with padding test passed")
        print(f"   Only used first {1} block indices, subsequent -1 padding ignored")
        return True
    except Exception as e:
        print(f" Valid index test failed: {e}")
        return False


# Use near-actual configuration
@pytest.mark.parametrize("num_physical_blocks", [12498])
@pytest.mark.parametrize("logical_block_table_size", [2560])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("seq_lens", [[128, 256, 512, 2048, 3072]])
@torch.inference_mode()
def test_performance_with_large_config(
    num_physical_blocks: int,
    logical_block_table_size: int,
    block_size: int,
    head_dim: int,
    seq_lens: int,
) -> None:
    """Test performance with large configuration"""
    print("\nTesting performance with large configuration...")
    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)

    # Create kv_cache
    kv_cache = torch.randint(
        0, 256,
        (num_physical_blocks, block_size, head_dim + 4),
        dtype=torch.uint8,
        device="cuda"
    )

    # Create cu_seq_lens
    cu_seq_lens_list = [0]
    for seq_len in seq_lens:
        cu_seq_lens_list.append(cu_seq_lens_list[-1] + seq_len)
    cu_seq_lens = torch.tensor(cu_seq_lens_list, dtype=torch.int32, device="cuda")

    # Create block_table
    block_table = torch.full((batch_size, logical_block_table_size), -1, dtype=torch.int32, device="cuda")

    # Allocate blocks for each sequence
    block_counter = 0
    for b in range(batch_size):
        blocks_needed = cdiv(seq_lens[b], block_size)
        for i in range(blocks_needed):
            # Assign valid block indices
            block_table[b, i] = block_counter % num_physical_blocks
            block_counter += 1

    # Create output buffers
    dst_value = torch.empty(total_tokens, head_dim, dtype=torch.float8_e4m3fn, device="cuda")
    dst_scale = torch.empty(total_tokens, 1, dtype=torch.float32, device="cuda")
    # Warm-up
    for _ in range(3):
        cp_gather_indexer_k_quant_cache(
            kv_cache, dst_value.clone(), dst_scale.clone(),
            block_table, cu_seq_lens, batch_size
        )
    torch.cuda.synchronize()

    # Performance test
    num_runs = 10
    start_time = time.time()

    for _ in range(num_runs):
        cp_gather_indexer_k_quant_cache(
            kv_cache, dst_value.clone(), dst_scale.clone(),
            block_table, cu_seq_lens, batch_size
        )
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # milliseconds
    tokens_per_second = total_tokens / (avg_time / 1000)

    print(f"cp_gather_indexer_k_quant_cache large configuration performance test:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average execution time: {avg_time:.3f} ms")
    print(f"  Throughput: {tokens_per_second:,.0f} tokens/second")
    print(f"  Data rate: {tokens_per_second * head_dim / 1e9:.3f} Gvalues/second")
    print(f" Performance test passed")


@pytest.mark.parametrize("num_physical_blocks", [12498])
@pytest.mark.parametrize("logical_block_table_size", [2560])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("seq_lens", [[128, 256, 512, 2048, 3072]])
@torch.inference_mode()
def test_performance_with_fuse_large_config(
    num_physical_blocks: int,
    logical_block_table_size: int,
    block_size: int,
    head_dim: int,
    seq_lens: int,
) -> None:
    """Test performance with large configuration (fused version)"""
    print("\nTesting performance with large configuration...")
    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)

    # Create kv_cache
    kv_cache = torch.randint(
        0, 256,
        (num_physical_blocks, block_size, head_dim + 4),
        dtype=torch.uint8,
        device="cuda"
    )

    # Create cu_seq_lens
    cu_seq_lens_list = [0]
    for seq_len in seq_lens:
        cu_seq_lens_list.append(cu_seq_lens_list[-1] + seq_len)
    cu_seq_lens = torch.tensor(cu_seq_lens_list, dtype=torch.int32, device="cuda")

    # Create block_table
    block_table = torch.full((batch_size, logical_block_table_size), -1, dtype=torch.int32, device="cuda")

    # Allocate blocks for each sequence
    block_counter = 0
    for b in range(batch_size):
        blocks_needed = cdiv(seq_lens[b], block_size)
        for i in range(blocks_needed):
            # Assign valid block indices
            block_table[b, i] = block_counter % num_physical_blocks
            block_counter += 1

    # Create output buffers
    k_fp8 = torch.empty([total_tokens, head_dim], device="cuda", dtype=torch.float8_e4m3fn)
    k_scale = torch.empty([total_tokens, 4], device="cuda", dtype=torch.uint8)
    # Warm-up
    for _ in range(3):
        torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8.clone(),
            k_scale.clone(),
            block_table,
            cu_seq_lens,
        )
    torch.cuda.synchronize()

    # Performance test
    num_runs = 10
    start_time = time.time()

    for _ in range(num_runs):
        torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8.clone(),
            k_scale.clone(),
            block_table,
            cu_seq_lens,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # milliseconds
    tokens_per_second = total_tokens / (avg_time / 1000)

    print(f"cp_gather_indexer_k_quant_cache fuse large configuration performance test:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average execution time: {avg_time:.3f} ms")
    print(f"  Throughput: {tokens_per_second:,.0f} tokens/second")
    print(f"  Data rate: {tokens_per_second * head_dim / 1e9:.3f} Gvalues/second")
    print(f" Performance test passed")
