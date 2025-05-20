/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "fused_add_rms_norm_static_int8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_add_rms_norm_static_int8_quant(
    at::Tensor& output, const at::Tensor& input, at::Tensor& residual,
    const at::Tensor& weight, double epsilon, const at::Tensor& scale) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor in_scale;
  if (scale.dim() == 0) {
    // per tensor
    in_scale = scale.reciprocal().to(input.dtype()).unsqueeze(0);
  } else {
    // per channel
    in_scale = scale.reciprocal().to(input.dtype());
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFusedAddRmsNormQuant)(
      output, input, const_cast<at::Tensor&>(residual), weight, epsilon,
      in_scale, stream));
}

}  // namespace vllm_gcu::llm_ops
