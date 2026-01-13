/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "top_k_per_row_prefill.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/tops/Context.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void top_k_per_row_prefill(const at::Tensor& logits,
                           const at::Tensor& rowStarts,
                           const at::Tensor& rowEnds, at::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(logits));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  if (numRows == 0) {
    return;
  }
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmTopkPerRowPrefill)(
      indices, logits, rowStarts, rowEnds, numRows, stride0, stride1, topK, -1,
      stream));
}

}  // namespace vllm_gcu::llm_ops
