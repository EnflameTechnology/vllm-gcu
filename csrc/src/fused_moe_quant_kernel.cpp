/*
 * Copyright 2024 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fused_moe_quant_kernel.h"

#include <c10/util/env.h>
#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void fused_moe_quant_kernel_gcu(
    at::Tensor &C, const at::Tensor &A, const at::Tensor &B,
    const c10::optional<at::Tensor> &A_scale, const at::Tensor &B_scale,
    int64_t gs, const c10::optional<at::Tensor> &B_zp,
    const at::Tensor &topk_weights, const at::Tensor &topk_ids,
    const at::Tensor &sorted_token_ids, const at::Tensor &experts_ids,
    const at::Tensor &num_tokens_post_pad, bool mul_routed_weight, int64_t topk,
    int64_t block_size, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &real_token_num) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(C));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor real_token_num_tensor;
  if (real_token_num.has_value()) {
    real_token_num_tensor = real_token_num.value();
  }
  at::Tensor bias_tensor;

  if (A_scale.has_value()) {  // w8a8
    at::Tensor A_scale_tensor = A_scale.value();
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
        topsvllm::topsvllmInvokeFusedMoeNonGatherQuantKernel)(
        C, A, B, A_scale_tensor, B_scale, bias_tensor, topk_weights, topk_ids,
        sorted_token_ids, experts_ids, num_tokens_post_pad,
        real_token_num_tensor, mul_routed_weight, topk, block_size, stream));
  } else {
    at::Tensor B_zp_tensor;
    if (B_zp.has_value()) {
      B_zp_tensor = B_zp.value();
    }
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
        topsvllm::topsvllmInvokeFusedMoeNonGatherQuantKernel)(
        C, A, B, B_scale, gs, B_zp_tensor, bias_tensor, topk_weights, topk_ids,
        sorted_token_ids, experts_ids, num_tokens_post_pad,
        real_token_num_tensor, mul_routed_weight, topk, block_size, stream));
  }
}

void fused_moe_quant_kernel(
    at::Tensor &C, const at::Tensor &A, const at::Tensor &B,
    const c10::optional<at::Tensor> &A_scale, const at::Tensor &B_scale,
    int64_t gs, const c10::optional<at::Tensor> &B_zp,
    const at::Tensor &topk_weights, const at::Tensor &topk_ids,
    const at::Tensor &sorted_token_ids, const at::Tensor &experts_ids,
    const at::Tensor &num_tokens_post_pad, bool mul_routed_weight, int64_t topk,
    int64_t block_size, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &real_token_num) {
  TORCH_CHECK(!bias.has_value(), "bias should be None.");

  // auto use_legacy = c10::utils::check_env("VLLM_FUSED_MOE_IMPL_LEGACY");

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor C_cpu, A_cpu, B_cpu, B_scale_cpu, topk_weights_cpu, topk_ids_cpu;
  at::Tensor sorted_token_ids_cpu, experts_ids_cpu, num_tokens_post_pad_cpu;
  at::Tensor A_scale_cpu, real_token_num_cpu, bias_cpu, B_zp_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("fused_moe_quant_kernel") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      C_cpu = C.to(at::kCPU);
      A_cpu = A.to(at::kCPU);
      B_cpu = B.to(at::kCPU);
      B_scale_cpu = B_scale.to(at::kCPU);
      topk_weights_cpu = topk_weights.to(at::kCPU);
      topk_ids_cpu = topk_ids.to(at::kCPU);
      sorted_token_ids_cpu = sorted_token_ids.to(at::kCPU);
      experts_ids_cpu = experts_ids.to(at::kCPU);
      num_tokens_post_pad_cpu = num_tokens_post_pad.to(at::kCPU);

      // Handle optional tensors
      if (A_scale.has_value()) {
        A_scale_cpu = A_scale.value().to(at::kCPU);
      }
      if (real_token_num.has_value()) {
        real_token_num_cpu = real_token_num.value().to(at::kCPU);
      } else {
        real_token_num_cpu = at::Tensor();
      }
      if (B_zp.has_value()) {
        B_zp_cpu = B_zp.value().to(at::kCPU);
      } else {
        B_zp_cpu = at::Tensor();
      }
      bias_cpu = at::Tensor();  // bias is checked to be None

      // Call native implementation on CPU tensors
      if (A_scale.has_value()) {
        // w8a8 path - use second function signature
        vllmInvokeFusedMoeNonGatherQuantKernel(
            C_cpu, A_cpu, B_cpu, A_scale_cpu, B_scale_cpu, bias_cpu,
            topk_weights_cpu, topk_ids_cpu, sorted_token_ids_cpu,
            experts_ids_cpu, num_tokens_post_pad_cpu, real_token_num_cpu,
            mul_routed_weight, topk, block_size);
      } else {
        // non-w8a8 path - use first function signature
        vllmInvokeFusedMoeNonGatherQuantKernel(
            C_cpu, A_cpu, B_cpu, B_scale_cpu, gs, B_zp_cpu, bias_cpu,
            topk_weights_cpu, topk_ids_cpu, sorted_token_ids_cpu,
            experts_ids_cpu, num_tokens_post_pad_cpu, real_token_num_cpu,
            mul_routed_weight, topk, block_size);
      }
    }
  }
#endif

  fused_moe_quant_kernel_gcu(C, A, B, A_scale, B_scale, gs, B_zp, topk_weights,
                             topk_ids, sorted_token_ids, experts_ids,
                             num_tokens_post_pad, mul_routed_weight, topk,
                             block_size, bias, real_token_num);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(C_cpu);
    auto device_outputs = std::make_tuple(C.to(at::kCPU));
    EXPECT_TRUE(
        vllmInvokeFusedMoeNonGatherQuantKernelCheck(cpu_output, device_outputs),
        "fused_moe_quant_kernel");
    C.copy_(C_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
