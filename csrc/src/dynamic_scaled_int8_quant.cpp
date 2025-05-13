/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "dynamic_scaled_int8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void dynamic_scaled_int8_quant(at::Tensor& output,
                               const at::Tensor& input,
                               at::Tensor& scales,
                               const at::Tensor& azp) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmDynamicScaledInt8Quant)(
      output, scales, input, azp, stream));
}

}  // namespace vllm_gcu::llm_ops
