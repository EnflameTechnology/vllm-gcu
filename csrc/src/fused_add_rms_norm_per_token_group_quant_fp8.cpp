/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "fused_add_rms_norm_per_token_group_quant_fp8.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_add_rms_norm_per_token_group_quant_fp8(
    at::Tensor &out, at::Tensor &residual, at::Tensor &scale,
    const at::Tensor &input, const at::Tensor &weight, double epsilon,
    int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  if (input.numel() == 0) return;

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmFusedAddRmsNormPerTokenGroupQuantFp8)(
          out, residual, scale, input, const_cast<at::Tensor &>(residual),
          weight, epsilon, group_size, stream));
}

}  // namespace vllm_gcu::llm_ops
