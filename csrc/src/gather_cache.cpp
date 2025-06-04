/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "gather_cache.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void gather_cache(const at::Tensor& src_cache, const at::Tensor& dst,
                  const at::Tensor& block_table, const at::Tensor& cu_seq_lens,
                  int64_t batch_size, const at::Tensor& seq_starts) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(dst));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGatherCache)(
      src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts,
      stream));
}

}  // namespace vllm_gcu::llm_ops
