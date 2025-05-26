/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "merge_attn_states.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void merge_attn_states(at::Tensor& output, at::Tensor& output_lse,
                     const at::Tensor& prefix_output,
                     const at::Tensor& prefix_lse,
                     const at::Tensor& suffix_output,
                     const at::Tensor& suffix_lse) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmMergeAttnStates)(
      output, output_lse, prefix_output, prefix_lse,
      suffix_output, suffix_lse, stream));
}

}  // namespace vllm_gcu::llm_ops
