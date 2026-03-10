/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "cutlass_scaled_mm.h"

#include <topsaten/topsaten_vllm.h>
#include <topsaten/topsaten_extensions.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void cutlass_scaled_mm(at::Tensor& out, const at::Tensor& x,
                       const at::Tensor& weight, const at::Tensor& x_scale,
                       const at::Tensor& w_scale,
                       const c10::optional<at::Tensor>& bias) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }

  if (weight.dtype() == torch::kFloat8_e4m3fn) {

    topsexts::topsextsScalingMode_t AScalingMode =
        topsexts::TOPSEXTS_INVALID_SCALING;
    topsexts::topsextsScalingMode_t BScalingMode =
        topsexts::TOPSEXTS_INVALID_SCALING;
    int group_size = -1;

    // Determine scaling modes
    // Add other scaling modes here in the future
    if (x_scale.dim() == 0 || (x_scale.dim() == 1 && x_scale.size(0) == 1)) {
      AScalingMode = topsexts::TOPSEXTS_SCALING_PER_TENSOR;
    }

    if (w_scale.dim() == 0 || (w_scale.dim() == 1 && w_scale.size(0) == 1)) {
      BScalingMode = topsexts::TOPSEXTS_SCALING_PER_TENSOR;
    }

    TORCH_CHECK(
        AScalingMode != topsexts::TOPSEXTS_INVALID_SCALING,
        "Unsupported activation scaling mode. "
        "x_scale shape: ", x_scale.sizes());

    TORCH_CHECK(
        BScalingMode != topsexts::TOPSEXTS_INVALID_SCALING,
        "Unsupported weight scaling mode. "
        "w_scale shape: ", w_scale.sizes());

    at::Tensor x_scale_modified = x_scale.to(torch::kFloat32);
    if (x_scale_modified.dim() == 0) {
      x_scale_modified = x_scale_modified.unsqueeze(0);
    }

    at::Tensor w_scale_modified = w_scale.to(torch::kFloat32);
    if (w_scale_modified.dim() == 0) {
      w_scale_modified = w_scale_modified.unsqueeze(0);
    }

    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsLinearQuant)(
        out, x, weight, bias_tensor, x_scale_modified, w_scale_modified,
        AScalingMode, BScalingMode, group_size, stream));
  } else {
    at::Tensor x_scale_modified = x_scale;
    if (x_scale.dim() == 0) {
      x_scale_modified = x_scale.unsqueeze(0);
    }

    // w_scale squeeze here
    at::Tensor w_scale_modified = w_scale;
    if (w_scale.dim() > 1) {
      w_scale_modified = w_scale.squeeze(-1);
    }

    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenDotBiasQuant)(
        out, x, weight, x_scale_modified, w_scale_modified,
        bias_tensor, stream));
  }
}

}  // namespace vllm_gcu::llm_ops
