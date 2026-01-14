# SPDX-License-Identifier: Apache-2.0
"""
测试 chunk_gated_delta_rule_fwd_h 函数
参数构造方法参考了 flash-linear-attention 官方测试：
https://github.com/fla-org/flash-linear-attention/blob/1d48579/tests/ops/test_gated_delta.py

"""
import pytest
import torch
import torch.nn.functional as F
import torch_gcu
from torch_gcu import transfer_to_gcu
from typing import Optional


from vllm_gcu.models.qwen3_next.chunk_delta_h import chunk_gated_delta_rule_fwd_h

# 导入参考实现需要的依赖
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices, prepare_chunk_offsets
from vllm.model_executor.layers.fla.ops.op import exp
from vllm.model_executor.layers.fla.ops.utils import is_nvidia_hopper, use_cuda_graph

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]

# ============================================================================
# 参考实现（使用 2D grid: N*H）
# ============================================================================
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4] for num_stages in [2, 3, 4] for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_ref(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * Hg + i_h // (H // Hg)) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV),
                                   (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV),
                                       (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV),
                                       (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV),
                                       (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1),
                                 (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1),
                                     (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2,
                     b_h2.to(p_h2.dtype.element_ty),
                     boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1),
                                     (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3,
                     b_h3.to(p_h3.dtype.element_ty),
                     boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1),
                                     (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4,
                     b_h4.to(p_h4.dtype.element_ty),
                     boundary_check=(0, 1))

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV),
                                (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1),
                                    (i_t * BT, i_v * BV), (BT, BV),
                                    (1, 0)) if SAVE_NEW_VALUE else None
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0),
                                (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64),
                                    (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128),
                                    (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192),
                                    (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1),
                                        (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new,
                     b_v_new.to(p_v_new.dtype.element_ty),
                     boundary_check=(0, 1))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T, ), (H, ),
                                    (i_t * BT, ), (BT, ), (0, ))
            b_g = tl.load(p_g, boundary_check=(0, ))
            b_v_new = b_v_new * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT),
                                (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT),
                                    (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT),
                                    (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT),
                                    (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v_new)

    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV),
                                 (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV),
                                     (64, BV), (1, 0))
            tl.store(p_ht,
                     b_h2.to(p_ht.dtype.element_ty),
                     boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV),
                                     (64, BV), (1, 0))
            tl.store(p_ht,
                     b_h3.to(p_ht.dtype.element_ty),
                     boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV),
                                     (64, BV), (1, 0))
            tl.store(p_ht,
                     b_h4.to(p_ht.dtype.element_ty),
                     boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_ref(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """参考实现：使用 2D grid (N*H)"""
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(
        cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(
            chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(
        N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_ref[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens.to(torch.int32) if cu_seqlens is not None else None,
        chunk_offsets=chunk_offsets.to(torch.int32) if chunk_offsets is not None else None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT)
    return h, v_new, final_state


# ============================================================================
# 测试函数
# ============================================================================
@pytest.mark.parametrize("B", [1])  
@pytest.mark.parametrize("T", [2048, 1212, 987])  
@pytest.mark.parametrize("H", [4])  
@pytest.mark.parametrize("Hg", [2])  
@pytest.mark.parametrize("K", [128])  
@pytest.mark.parametrize("V", [128])  
@pytest.mark.parametrize("dtype", [torch.bfloat16])  
@pytest.mark.parametrize("use_g", [True])  
@pytest.mark.parametrize("use_initial_state", [True])  
@pytest.mark.parametrize("output_final_state", [True])  
def test_chunk_gated_delta_rule_fwd_h_accuracy(B, T, H, Hg, K, V, dtype, use_g,
                                                 use_initial_state,
                                                 output_final_state):
    """
    精度测试：比较新实现 (grid=(N, H)) 和参考实现 (grid=(N*H)) 的输出精度
    测试场景：固定长度序列
    """
    print(f"Testing accuracy test (fixed length)...")
    if K > 256:
        pytest.skip("K > 256 is not supported")
    
    if H % Hg != 0:
        pytest.skip("H must be divisible by Hg")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    
    device = torch.device("cuda")
    print(f"device: {device}")
    chunk_size = 64
    
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, Hg, K, dtype=torch.float32, device=device), 
                    p=2, dim=-1).to(dtype)
    u = torch.randn(B, T, H, V, dtype=dtype, device=device)
    w = torch.rand(B, T, H, K, dtype=dtype, device=device).sigmoid()
    
    if use_g:
        g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device))
    else:
        g = None
    
    initial_state = torch.randn(
        B, H, K, V, dtype=torch.float32,
        device=device) if use_initial_state else None
    
    # 调用新实现
    h_new, v_new_new, final_state_new = chunk_gated_delta_rule_fwd_h(
        k=k.clone(),
        w=w.clone(),
        u=u.clone(),
        g=g.clone() if g is not None else None,
        initial_state=initial_state.clone() if initial_state is not None else None,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=None,
    )
    
    # 调用参考实现
    h_ref, v_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h_ref(
        k=k.clone(),
        w=w.clone(),
        u=u.clone(),
        g=g.clone() if g is not None else None,
        initial_state=initial_state.clone() if initial_state is not None else None,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=None,
    )
    
    rtol = 1e-4 
    atol = 1e-4 
    
    # 比较 h
    print("\n Comparing h tensors...")
    diff = torch.abs(h_new - h_ref)
    h_max_diff = diff.max().item()
    h_mean_diff = diff.mean().item()
    h_median_diff = diff.median().item()
    
    h_ref_abs = torch.abs(h_ref)
    rel_diff = diff / (h_ref_abs + 1e-8)
    h_max_rel_diff = rel_diff.max().item()
    h_mean_rel_diff = rel_diff.mean().item()
    
    print(f"  Absolute diff: max={h_max_diff:.6e}, mean={h_mean_diff:.6e}, median={h_median_diff:.6e}")
    print(f"  Relative diff: max={h_max_rel_diff:.6e}, mean={h_mean_rel_diff:.6e}")
    
    h_match = torch.allclose(h_new, h_ref, rtol=rtol, atol=atol)
    assert h_match, (f"h tensors do not match: "
                     f"max_abs_diff={h_max_diff:.6e}, max_rel_diff={h_max_rel_diff:.6e}, "
                     f"required: rtol={rtol}, atol={atol}")
    
    # 比较 v_new
    print("\n Comparing v_new tensors...")
    v_diff = torch.abs(v_new_new - v_new_ref)
    v_max_diff = v_diff.max().item()
    v_mean_diff = v_diff.mean().item()
    v_median_diff = v_diff.median().item()
    
    v_ref_abs = torch.abs(v_new_ref)
    v_rel_diff = v_diff / (v_ref_abs + 1e-8)
    v_max_rel_diff = v_rel_diff.max().item()
    v_mean_rel_diff = v_rel_diff.mean().item()
    
    print(f"  Absolute diff: max={v_max_diff:.6e}, mean={v_mean_diff:.6e}, median={v_median_diff:.6e}")
    print(f"  Relative diff: max={v_max_rel_diff:.6e}, mean={v_mean_rel_diff:.6e}")
    
    v_new_match = torch.allclose(v_new_new, v_new_ref, rtol=rtol, atol=atol)
    assert v_new_match, (f"v_new tensors do not match: "
                         f"max_abs_diff={v_max_diff:.6e}, max_rel_diff={v_max_rel_diff:.6e}, "
                         f"required: rtol={rtol}, atol={atol}")
    
    # 比较 final_state
    if output_final_state:
        print("\n Comparing final_state tensors...")
        fs_diff = torch.abs(final_state_new - final_state_ref)
        fs_max_diff = fs_diff.max().item()
        fs_mean_diff = fs_diff.mean().item()
        fs_median_diff = fs_diff.median().item()
        
        fs_ref_abs = torch.abs(final_state_ref)
        fs_rel_diff = fs_diff / (fs_ref_abs + 1e-8)
        fs_max_rel_diff = fs_rel_diff.max().item()
        fs_mean_rel_diff = fs_rel_diff.mean().item()
        
        print(f"  Absolute diff: max={fs_max_diff:.6e}, mean={fs_mean_diff:.6e}, median={fs_median_diff:.6e}")
        print(f"  Relative diff: max={fs_max_rel_diff:.6e}, mean={fs_mean_rel_diff:.6e}")
        
        final_state_match = torch.allclose(final_state_new, final_state_ref, rtol=rtol, atol=atol)
        assert final_state_match, (f"final_state tensors do not match: "
                                   f"max_abs_diff={fs_max_diff:.6e}, max_rel_diff={fs_max_rel_diff:.6e}, "
                                   f"required: rtol={rtol}, atol={atol}")
    
    print(f"Accuracy test passed: B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, "
          f"dtype={dtype}, use_g={use_g}, use_initial_state={use_initial_state}, "
          f"output_final_state={output_final_state}")


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [2048])  
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("Hg", [2])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("V", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_gated_delta_rule_fwd_h_accuracy_varlen(B, T, H, Hg, K, V, dtype):
    """
    精度测试（变长序列）：比较新实现和参考实现在变长序列上的输出精度
    使用示例的cu_seqlens 配置
    """
    print(f"Testing accuracy test (varlen)...")
    if K > 256:
        pytest.skip("K > 256 is not supported")
    
    if H % Hg != 0:
        pytest.skip("H must be divisible by Hg")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    
    device = torch.device("cuda")
    print(f"device: {device}")
    chunk_size = 64
    
    # 创建示例的变长序列的 cu_seqlens
    # 12个序列：长度分别为 1,1,1,1,1,1,1,1,1,46,1024,969
    cu_seqlens = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 55, 1079, 2048],
        dtype=torch.int32,
        device=device
    )
    num_seqs = len(cu_seqlens) - 1  # 12个序列
    
    print(f"  T={T}, H={H}, Hg={Hg}, K={K}, V={V}")
    print(f"  cu_seqlens: {cu_seqlens.tolist()}")
    print(f"  num_sequences: {num_seqs}")
    
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, Hg, K, dtype=torch.float32, device=device), 
                    p=2, dim=-1).to(dtype)
    u = torch.randn(B, T, H, V, dtype=dtype, device=device)
    w = torch.rand(B, T, H, K, dtype=dtype, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device))
    # initial_state: [num_seqs, H, K, V] - 每个序列一个初始状态
    initial_state = torch.randn(num_seqs, H, K, V, dtype=torch.float32, device=device)
    
    print("\n Calling NEW implementation (3D grid)...")
    h_new, v_new_new, final_state_new = chunk_gated_delta_rule_fwd_h(
        k=k.clone(),
        w=w.clone(),
        u=u.clone(),
        g=g.clone(),
        initial_state=initial_state.clone(),
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens.clone(),
    )
    print(f"  h_new: shape={h_new.shape}, dtype={h_new.dtype}")
    print(f"  h_new stats: min={h_new.min().item():.6e}, max={h_new.max().item():.6e}, mean={h_new.mean().item():.6e}")
    print(f"  h_new has NaN: {torch.isnan(h_new).any().item()}, Inf: {torch.isinf(h_new).any().item()}")
    
    print("\n Calling REF implementation (2D grid)...")
    h_ref, v_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h_ref(
        k=k.clone(),
        w=w.clone(),
        u=u.clone(),
        g=g.clone(),
        initial_state=initial_state.clone(),
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens.clone(),
    )
    print(f"  h_ref: shape={h_ref.shape}, dtype={h_ref.dtype}")
    print(f"  h_ref stats: min={h_ref.min().item():.6e}, max={h_ref.max().item():.6e}, mean={h_ref.mean().item():.6e}")
    print(f"  h_ref has NaN: {torch.isnan(h_ref).any().item()}, Inf: {torch.isinf(h_ref).any().item()}")
    
    rtol = 1e-4 
    atol = 1e-4   
    
    print("\n Comparing h tensors...")
    diff = torch.abs(h_new - h_ref)
    h_max_diff = diff.max().item()
    h_mean_diff = diff.mean().item()
    h_median_diff = diff.median().item()
    
    # 计算相对误差
    h_ref_abs = torch.abs(h_ref)
    rel_diff = diff / (h_ref_abs + 1e-8)
    h_max_rel_diff = rel_diff.max().item()
    h_mean_rel_diff = rel_diff.mean().item()
    
    print(f"  Absolute diff: max={h_max_diff:.6e}, mean={h_mean_diff:.6e}, median={h_median_diff:.6e}")
    print(f"  Relative diff: max={h_max_rel_diff:.6e}, mean={h_mean_rel_diff:.6e}")
    print(f"  h_ref magnitude: max={h_ref_abs.max().item():.6e}, mean={h_ref_abs.mean().item():.6e}")
    
    h_match = torch.allclose(h_new, h_ref, rtol=rtol, atol=atol)
    if not h_match:
        print(f"\n h mismatch: max_diff={h_max_diff:.6e}, mean_diff={h_mean_diff:.6e}")
        print(f"   Required: rtol={rtol}, atol={atol}")
        print(f"   Suggestion: Use rtol=1e-2, atol=1e6 for varlen tests (relative error ~{h_max_rel_diff:.2e})")
    assert h_match, (f"h tensors do not match: "
                     f"max_abs_diff={h_max_diff:.6e}, max_rel_diff={h_max_rel_diff:.6e}, "
                     f"required: rtol={rtol}, atol={atol}")
    
    # 比较 v_new
    print("\n Comparing v_new tensors...")
    v_diff = torch.abs(v_new_new - v_new_ref)
    v_max_diff = v_diff.max().item()
    v_mean_diff = v_diff.mean().item()
    v_median_diff = v_diff.median().item()
    
    # 计算相对误差
    v_ref_abs = torch.abs(v_new_ref)
    v_rel_diff = v_diff / (v_ref_abs + 1e-8)
    v_max_rel_diff = v_rel_diff.max().item()
    v_mean_rel_diff = v_rel_diff.mean().item()
    
    print(f"  Absolute diff: max={v_max_diff:.6e}, mean={v_mean_diff:.6e}, median={v_median_diff:.6e}")
    print(f"  Relative diff: max={v_max_rel_diff:.6e}, mean={v_mean_rel_diff:.6e}")
    print(f"  v_new_ref magnitude: max={v_ref_abs.max().item():.6e}, mean={v_ref_abs.mean().item():.6e}")
    
    v_new_match = torch.allclose(v_new_new, v_new_ref, rtol=rtol, atol=atol)
    if not v_new_match:
        print(f"\n v_new mismatch: max_diff={v_max_diff:.6e}, mean_diff={v_mean_diff:.6e}")
        print(f"   Required: rtol={rtol}, atol={atol}")
        print(f"   Suggestion: Use rtol=1e-2, atol=1e6 for varlen tests (relative error ~{v_max_rel_diff:.2e})")
    assert v_new_match, (f"v_new tensors do not match: "
                         f"max_abs_diff={v_max_diff:.6e}, max_rel_diff={v_max_rel_diff:.6e}, "
                         f"required: rtol={rtol}, atol={atol}")
    
    # 比较 final_state
    print("\n Comparing final_state tensors...")
    fs_diff = torch.abs(final_state_new - final_state_ref)
    fs_max_diff = fs_diff.max().item()
    fs_mean_diff = fs_diff.mean().item()
    fs_median_diff = fs_diff.median().item()
    
    # 计算相对误差
    fs_ref_abs = torch.abs(final_state_ref)
    fs_rel_diff = fs_diff / (fs_ref_abs + 1e-8)
    fs_max_rel_diff = fs_rel_diff.max().item()
    fs_mean_rel_diff = fs_rel_diff.mean().item()
    
    print(f"  Absolute diff: max={fs_max_diff:.6e}, mean={fs_mean_diff:.6e}, median={fs_median_diff:.6e}")
    print(f"  Relative diff: max={fs_max_rel_diff:.6e}, mean={fs_mean_rel_diff:.6e}")
    print(f"  final_state_ref magnitude: max={fs_ref_abs.max().item():.6e}, mean={fs_ref_abs.mean().item():.6e}")
    
    final_state_match = torch.allclose(final_state_new, final_state_ref, rtol=rtol, atol=atol)
    if not final_state_match:
        print(f"\n final_state mismatch: max_diff={fs_max_diff:.6e}, mean_diff={fs_mean_diff:.6e}")
        print(f"   Required: rtol={rtol}, atol={atol}")
        print(f"   Suggestion: Use rtol=1e-2, atol=1e6 for varlen tests (relative error ~{fs_max_rel_diff:.2e})")
    assert final_state_match, (f"final_state tensors do not match: "
                               f"max_abs_diff={fs_max_diff:.6e}, max_rel_diff={fs_max_rel_diff:.6e}, "
                               f"required: rtol={rtol}, atol={atol}")
    
    print(f"Varlen accuracy test passed: B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, dtype={dtype}")


if __name__ == "__main__": 
    
    print("\n" + "=" * 80)
    print("Testing Accuracy (New vs Reference Implementation)")
    print("=" * 80)
    
    print("\n1. Accuracy test with T=1212...")
    test_chunk_gated_delta_rule_fwd_h_accuracy(
        B=1,
        T=1212,
        H=4,
        Hg=2,
        K=128,
        V=128,
        dtype=torch.bfloat16,
        use_g=True,
        use_initial_state=True,
        output_final_state=True,
    )
    
    print("\n2. Varlen accuracy test with T=2048...")
    test_chunk_gated_delta_rule_fwd_h_accuracy_varlen(
        B=1,
        T=2048,
        H=4,
        Hg=2,
        K=128,
        V=128,
        dtype=torch.bfloat16,
    )
    
    print("\n" + "=" * 80)
    print("All core tests passed!")
    print("Note: Varlen accuracy test uses production cu_seqlens configuration")
    print("=" * 80)
