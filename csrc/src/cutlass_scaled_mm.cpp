/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "cutlass_scaled_mm.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void cutlass_scaled_mm(at::Tensor& out, const at::Tensor& x,
                      const at::Tensor& weight,
                      const at::Tensor& x_scale,
                      const at::Tensor& w_scale,
                      const c10::optional<at::Tensor>& bias) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }

  at::Tensor x_scale_modified = x_scale;
  if (x_scale.dim() == 0) {
    x_scale_modified = x_scale.unsqueeze(0);
  }

  // w_scale squeeze here
  at::Tensor w_scale_modified = w_scale.squeeze(-1);

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenDotBiasQuant)(
      out, x, weight, x_scale_modified, w_scale_modified, bias_tensor, stream));
}

}  // namespace vllm_gcu::llm_ops
