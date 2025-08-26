/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "fused_moe_quant_kernel_ex.h"

#include <topsaten/topsaten_vllm.h>

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

} // namespace vllm_gcu::llm_ops
