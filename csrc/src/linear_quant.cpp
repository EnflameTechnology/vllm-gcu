/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "linear_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void linear_quant(at::Tensor &out, const at::Tensor &lhs,
                 const at::Tensor &rhs, const c10::optional<at::Tensor> &bias,
                 const at::Tensor &lhs_scale,
                 const at::Tensor &rhs_scale) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  if (lhs.numel() == 0) return;

  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsaten::topsatenLinearQuant)(
          out, lhs, rhs, bias_tensor, lhs_scale, rhs_scale, stream));
}

} // namespace vllm_gcu::llm_ops
