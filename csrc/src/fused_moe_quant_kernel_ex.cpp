/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "fused_moe_quant_kernel_ex.h"

#include <topsaten/topsaten_vllm.h>
#include <topsaten/topsaten_extensions.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void fused_moe_quant_kernel_ex(
    at::Tensor &C, const at::Tensor &A, const at::Tensor &B,
    const at::Tensor &A_scale, const at::Tensor &B_scale,
    const at::Tensor &B_zero, const c10::optional<at::Tensor> &bias,
    const at::Tensor &topk_weights, const at::Tensor &topk_ids,
    const at::Tensor &sorted_token_ids, const at::Tensor &experts_ids,
    const at::Tensor &num_tokens_post_pad,
    const c10::optional<at::Tensor> &real_token_num,
    bool mul_routed_weight, int64_t topk, int64_t block_size,
    int64_t group_k, int64_t group_n) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(C));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }
  at::Tensor real_token_num_tensor;
  if (real_token_num.has_value()) {
    real_token_num_tensor = real_token_num.value();
  }

  if (B.dtype() == torch::kFloat8_e4m3fn) {

    topsexts::topsextsScalingMode_t AScalingMode =
        topsexts::TOPSEXTS_INVALID_SCALING;
    topsexts::topsextsScalingMode_t BScalingMode =
        topsexts::TOPSEXTS_INVALID_SCALING;

    // Determine scaling modes
    // Add other scaling modes here in the future
    if (A_scale.dim() == 0 || (A_scale.dim() == 1 && A_scale.size(0) == 1)) {
      AScalingMode = topsexts::TOPSEXTS_SCALING_PER_TENSOR;
    }

    if (B_scale.dim() == 1 || (B_scale.dim() == 2 && B_scale.size(1) == 1)) {
      BScalingMode = topsexts::TOPSEXTS_SCALING_PER_TENSOR;
    }

    TORCH_CHECK(
      AScalingMode != topsexts::TOPSEXTS_INVALID_SCALING,
      "Unsupported activation scaling mode. "
      "A_scale shape: ", A_scale.sizes());

    TORCH_CHECK(
      BScalingMode != topsexts::TOPSEXTS_INVALID_SCALING,
      "Unsupported weight scaling mode. "
      "B_scale shape: ", B_scale.sizes());

    at::Tensor A_scale_modified = A_scale.to(torch::kFloat32);
    if (A_scale_modified.dim() == 0) {
      A_scale_modified = A_scale_modified.unsqueeze(0);
    }

    at::Tensor B_scale_modified = B_scale.to(torch::kFloat32);
    if (B_scale_modified.dim() == 1) {
      B_scale_modified = B_scale_modified.unsqueeze(-1);
    }

    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsInvokeFusedMoeKernel)(
        C, A, B,
        A_scale_modified, B_scale_modified, B_zero, bias_tensor,
        topk_weights,
        sorted_token_ids, experts_ids,
        num_tokens_post_pad, real_token_num_tensor,
        mul_routed_weight,
        static_cast<int>(topk), static_cast<int>(block_size),
        static_cast<int>(group_k), static_cast<int>(group_n),
        AScalingMode, BScalingMode,
        stream));
  } else {
    // Turn B from [e, n, k] -> [e, k, n] by view
    at::Tensor B_reshaped = B.view({B.size(0), B.size(2), B.size(1)});

    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
        topsvllm::topsvllmInvokeFusedMoeNonGatherQuantKernel)(
        C, A, B_reshaped, A_scale, B_scale, B_zero, bias_tensor,
        topk_weights, topk_ids, sorted_token_ids, experts_ids,
        num_tokens_post_pad, real_token_num_tensor,
        mul_routed_weight, static_cast<int>(topk), static_cast<int>(block_size),
        static_cast<int>(group_k), static_cast<int>(group_n), stream));
  }
}

} // namespace vllm_gcu::llm_ops
