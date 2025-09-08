/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "expand_batch_to_tokens.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void expand_batch_to_tokens(at::Tensor &output, const at::Tensor &input,
                            const at::Tensor &cu_num_tokens, int64_t num_tokens,
                            double replace_from, double replace_to) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmExpandBatchtoTokens)(
      output, input, cu_num_tokens, num_tokens, replace_from, replace_to,
      stream));
}

} // namespace vllm_gcu::llm_ops
