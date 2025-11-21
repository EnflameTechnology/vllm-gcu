/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "dynamic_per_token_group_fp8_quant_with_ue8m0.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void dynamic_per_token_group_fp8_quant_with_ue8m0(
                                       at::Tensor &out,
                                       at::Tensor &scale,
                                       const at::Tensor &input,
                                       int64_t group_size,
                                       bool scale_to_ue8m0) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor out_fp32;

  if (input.numel() == 0) return;

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmDynamicPerTokenGroupFP8Quant)(
          out, scale, out_fp32, input, group_size, scale_to_ue8m0, stream));
}
}  // namespace vllm_gcu::llm_ops
