/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "fused_qk_norm_rope.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_qk_norm_rope(at::Tensor &qkv, int64_t num_heads_q,
                        int64_t num_heads_k, int64_t num_heads_v,
                        int64_t head_dim, double eps, at::Tensor &q_weight,
                        at::Tensor &k_weight, at::Tensor &cos_sin_cache,
                        bool is_neox, at::Tensor &position_ids) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(qkv));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
}
} // namespace vllm_gcu::llm_ops

