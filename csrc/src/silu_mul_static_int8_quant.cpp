/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "silu_mul_static_int8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void silu_mul_static_int8_quant(at::Tensor& result, const at::Tensor& input,
                                const at::Tensor& scale) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(result));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor in_scale;
  if (scale.dim() == 0) {
    // per tensor
    in_scale = scale.reciprocal().to(input.dtype()).unsqueeze(0);
  } else {
    // per channel
    in_scale = scale.reciprocal().to(input.dtype());
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmSiluMulQuant)(
      result, input, in_scale, stream));
}

}  // namespace vllm_gcu::llm_ops
