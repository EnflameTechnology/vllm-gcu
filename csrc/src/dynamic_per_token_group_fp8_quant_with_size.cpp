/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include <topsaten/topsaten_extensions.h>

#include "dynamic_per_token_group_fp8_quant.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void dynamic_per_token_group_fp8_quant_with_size(at::Tensor& output,
                                                 at::Tensor& scale,
                                                 const at::Tensor& input,
                                                 const at::Tensor& real_size,
                                                 int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor out_fp32;

  if (input.numel() == 0) return;

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsexts::topsextsDynamicPerTokenGroupFP8Quant)(
          output, scale, out_fp32, input, real_size, group_size, stream));
}

}  // namespace vllm_gcu::llm_ops
