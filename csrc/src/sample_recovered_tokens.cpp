/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "sample_recovered_tokens.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void sample_recovered_tokens(at::Tensor &output_token_ids,
                             const at::Tensor &cu_num_draft_tokens,
                             const at::Tensor &draft_token_ids,
                             const at::Tensor &target_probs,
                             const at::Tensor &q,
                             const c10::optional<at::Tensor> &draft_probs) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output_token_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  // Check if draft_probs is provided to decide which overload to call
  if (draft_probs.has_value()) {
    // Call the version with draft_probs
    at::Tensor draft_probs_tensor = draft_probs.value();
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmSampleRecoveredTokens)(
        output_token_ids, cu_num_draft_tokens, draft_token_ids,
        draft_probs_tensor, target_probs, q, stream));
  } else {
    // Call the version without draft_probs
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmSampleRecoveredTokens)(
        output_token_ids, cu_num_draft_tokens, draft_token_ids, target_probs, q,
        stream));
  }
}

} // namespace vllm_gcu::llm_ops
