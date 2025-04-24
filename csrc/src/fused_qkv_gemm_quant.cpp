/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "fused_qkv_gemm_quant.h"

#include <topsaten/topsaten_extensions.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_qkv_gemm_quant(at::Tensor &q, at::Tensor &kv, const at::Tensor &x,
                          const at::Tensor &weight, const at::Tensor &scale,
                          const at::Tensor &zeros, int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(q));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsFusedQKVGemmQuant)(
      q, kv, x, weight, scale, zeros, group_size, stream));
}

}  // namespace vllm_gcu::llm_ops
