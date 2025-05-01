/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "ets_moe_align_block_size.h"

#include <topsaten/topsaten_extensions.h>
#include <vector>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void ets_moe_align_block_size(at::Tensor& sorted_token_ids,
                          at::Tensor& experts_ids,
                          at::Tensor& num_tokens_post_pad,
                          const at::Tensor& topk_ids,
                          const at::Tensor& real_token_num,
                          const at::Tensor& expert_map,
                          int64_t num_experts,
                          int64_t block_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(sorted_token_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  // 直接进行参数传递和类型转换
  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsexts::topsextsMoeAlignBlockSize)(
          sorted_token_ids, experts_ids, num_tokens_post_pad,
          topk_ids, real_token_num, expert_map,
          static_cast<int>(num_experts),
          static_cast<int>(block_size),
          stream));
}

} // namespace vllm_gcu::llm_ops
