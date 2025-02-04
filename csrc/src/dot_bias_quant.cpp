/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "dot_bias_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void dot_bias_quant(at::Tensor& out, const at::Tensor& lhs,
                    const at::Tensor& rhs, const at::Tensor& scale,
                    const c10::optional<at::Tensor>& bias) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value().to(at::kFloat);
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenDotBiasQuant)(
      out, lhs, rhs, scale, bias_tensor, stream));
}

}  // namespace vllm_gcu::llm_ops
