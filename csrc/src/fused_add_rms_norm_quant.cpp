/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "fused_add_rms_norm_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_add_rms_norm_quant(at::Tensor& output, const at::Tensor& input,
                              const at::Tensor& residual,
                              const at::Tensor& weight, double epsilon,
                              const at::Tensor& scaling) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFusedAddRmsNormQuant)(
      output, input, residual, weight, epsilon, scaling, stream));
}

}  // namespace vllm_gcu::llm_ops
