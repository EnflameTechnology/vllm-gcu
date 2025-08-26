/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "mul_static_fp8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void mul_static_fp8_quant(at::Tensor &out, const at::Tensor &input,
                          const at::Tensor &scale,
                          const at::Tensor &real_num_tokens) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor smooth_scale;
  int group_size = -1;
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmMulStaticFp8Quant)(
      out, input, scale, real_num_tokens, smooth_scale, static_cast<int>(0),
      group_size, stream));
}

} // namespace vllm_gcu::llm_ops
