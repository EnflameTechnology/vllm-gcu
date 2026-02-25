/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "cp_gather_cache.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void cp_gather_cache(const at::Tensor& src_cache, at::Tensor& dst,
                     const at::Tensor& block_table,
                     const at::Tensor& cu_seq_lens, int64_t batch_size,
                     const ::std::optional<at::Tensor>& seq_starts) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(dst));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::Tensor seq_starts_tensor;
  if (seq_starts.has_value()) {
    seq_starts_tensor = seq_starts.value();
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGatherCache)(
      src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts_tensor,
      stream));
}

}  // namespace vllm_gcu::llm_ops
