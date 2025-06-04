/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "gelu_tanh_static_int8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void gelu_tanh_static_int8_quant(at::Tensor& out, const at::Tensor& input,
                                 const at::Tensor& scale) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGeluTanhQuant)(
      out, input, scale, stream));
}

}  // namespace vllm_gcu::llm_ops
