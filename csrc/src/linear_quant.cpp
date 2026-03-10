/**
* Copyright 2024 Enflame. All Rights Reserved.
*/
#include "linear_quant.h"
#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
  void linear_quant(at::Tensor &out,
    const at::Tensor &lhs,
    const at::Tensor &rhs,
    const c10::optional<at::Tensor> &bias,
    const at::Tensor &lhs_scale,
    const c10::optional<at::Tensor> &rhs_scale,
    int64_t group_size = -1) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  if (lhs.numel() == 0) return;
  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }

  // linear quant depends on rhs_scale
  if (rhs_scale.has_value()) {
    auto rhs_scale_tensor = rhs_scale.value();
    ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsaten::topsatenLinearQuant)(
        out, lhs, rhs, bias_tensor, lhs_scale, rhs_scale_tensor, stream));
  } else {
    ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsaten::topsatenLinearQuant)(
        out, lhs, rhs, bias_tensor, lhs_scale,
        static_cast<int>(group_size), stream));
  }

}

} // namespace vllm_gcu::llm_ops
