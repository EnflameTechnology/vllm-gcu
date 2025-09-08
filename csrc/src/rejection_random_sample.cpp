/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "rejection_random_sample.h"

#include <topsaten/topsaten_define.h>
#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void rejection_random_sample(
    at::Tensor &output_token_ids, const at::Tensor &cu_num_draft_tokens,
    const at::Tensor &draft_token_ids, const at::Tensor &draft_probs,
    const at::Tensor &target_probs, const at::Tensor &bonus_token_ids,
    const at::Tensor &recovered_token_ids, const at::Tensor &uniform_probs,
    const at::Tensor &is_greedy) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output_token_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  // Check if draft_probs is provided to decide which overload to call
  if (draft_probs.defined()) {
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmRejectionRandomSample)(
        output_token_ids, cu_num_draft_tokens, draft_token_ids, draft_probs,
        target_probs, bonus_token_ids, recovered_token_ids, uniform_probs,
        is_greedy, stream));
  } else {
    topsatenTensor draft_probs_tensor;
    draft_probs_tensor = topsatenTensor();
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmRejectionRandomSample)(
        output_token_ids, cu_num_draft_tokens, draft_token_ids,
        draft_probs_tensor, target_probs, bonus_token_ids, recovered_token_ids,
        uniform_probs, is_greedy, stream));
  }
}

} // namespace vllm_gcu::llm_ops
