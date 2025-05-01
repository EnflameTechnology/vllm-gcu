/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "fused_qkv_proj.h"

#include <topsaten/topsaten_extensions.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_qkv_proj(at::Tensor &q,
                     at::Tensor &kv,
                     const at::Tensor &x,
                     const at::Tensor &weight,
                     const at::Tensor &x_scale,
                     const at::Tensor &weight_scale,
                     int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(q));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsexts::topsextsFusedQKVProj)(
          q, kv, x, weight, x_scale, weight_scale, group_size, stream));
}

} // namespace vllm_gcu::llm_ops
