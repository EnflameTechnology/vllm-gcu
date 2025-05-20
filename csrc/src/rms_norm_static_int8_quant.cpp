/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "rms_norm_static_int8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void rms_norm_static_int8_quant(at::Tensor& output, const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& scale, double epsilon) {
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
  at::Scalar scalar_epsilon(epsilon);

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmRmsNormQuant)(
      output, input, weight, in_scale, scalar_epsilon, stream));
}

}  // namespace vllm_gcu::llm_ops
