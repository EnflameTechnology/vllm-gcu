/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "rejection_greedy_sample.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void rejection_greedy_sample(at::Tensor &output_token_ids,
                             const at::Tensor &cu_num_draft_tokens,
                             const at::Tensor &draft_token_ids,
                             const at::Tensor &target_argmax,
                             const at::Tensor &bonus_token_ids,
                             const at::Tensor &is_greedy) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output_token_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmRejectionGreedySample)(
      output_token_ids, cu_num_draft_tokens, draft_token_ids, target_argmax,
      bonus_token_ids, is_greedy, stream));
}

} // namespace vllm_gcu::llm_ops
