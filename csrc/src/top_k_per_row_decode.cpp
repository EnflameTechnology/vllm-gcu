/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "top_k_per_row_decode.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/tops/Context.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void top_k_per_row_decode(const at::Tensor& logits, int64_t next_n,
                          const at::Tensor& seq_lens, at::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK, int64_t threshold) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(logits));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  if (numRows == 0) {
    return;
  }
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmTopkPerRowDecode)(
      indices, logits, seq_lens, next_n, numRows, stride0, stride1, topK,
      threshold, stream));
}
}  // namespace vllm_gcu::llm_ops
