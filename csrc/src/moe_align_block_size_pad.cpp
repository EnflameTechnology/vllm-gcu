/*
 * Copyright 2025 Enflame. All Rights Reserved.

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

#include "moe_align_block_size_pad.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include <string>
#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void moe_align_block_size_pad_gcu(at::Tensor topk_ids, at::Tensor topk_ids_size,
                                  int64_t num_experts, int64_t block_size,
                                  at::Tensor sorted_token_ids,
                                  at::Tensor experts_ids,
                                  at::Tensor num_tokens_post_pad) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(topk_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmMoeAlignBlockSize)(
      sorted_token_ids, experts_ids, num_tokens_post_pad, topk_ids,
      topk_ids_size, num_experts, block_size, stream));
}

void moe_align_block_size_pad(at::Tensor topk_ids, at::Tensor topk_ids_size,
                              int64_t num_experts, int64_t block_size,
                              at::Tensor sorted_token_ids,
                              at::Tensor experts_ids,
                              at::Tensor num_tokens_post_pad) {
#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor topk_ids_cpu, topk_ids_size_cpu, sorted_token_ids_cpu,
      experts_ids_cpu, num_tokens_post_pad_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("moe_align_block_size_pad") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("moe_align_block_size_pad",
                            "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      topk_ids_cpu = topk_ids.to(at::kCPU);
      topk_ids_size_cpu = topk_ids_size.to(at::kCPU);
      sorted_token_ids_cpu = sorted_token_ids.to(at::kCPU);
      experts_ids_cpu = experts_ids.to(at::kCPU);
      num_tokens_post_pad_cpu = num_tokens_post_pad.to(at::kCPU);

      // Call native implementation on CPU tensors - using the version with
      // real_token_num
      int result = vllmMoeAlignBlockSize(
          sorted_token_ids_cpu, experts_ids_cpu, num_tokens_post_pad_cpu,
          topk_ids_cpu, topk_ids_size_cpu, static_cast<int>(num_experts),
          static_cast<int>(block_size));

      VLLM_FALLBACK_CPU_LOG("moe_align_block_size_pad",
                            "CPU fallback computation completed");
    }
  }
#endif

  moe_align_block_size_pad_gcu(topk_ids, topk_ids_size, num_experts, block_size,
                               sorted_token_ids, experts_ids,
                               num_tokens_post_pad);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("moe_align_block_size_pad",
                          "Starting result verification");

    auto cpu_output = std::make_tuple(sorted_token_ids_cpu, experts_ids_cpu,
                                      num_tokens_post_pad_cpu);
    auto device_outputs =
        std::make_tuple(sorted_token_ids.to(at::kCPU), experts_ids.to(at::kCPU),
                        num_tokens_post_pad.to(at::kCPU));
    EXPECT_TRUE(vllmMoeAlignBlockSizeCheck(cpu_output, device_outputs),
                "moe_align_block_size_pad");
    sorted_token_ids.copy_(sorted_token_ids_cpu);
    experts_ids.copy_(experts_ids_cpu);
    num_tokens_post_pad.copy_(num_tokens_post_pad_cpu);

    VLLM_FALLBACK_CPU_LOG("moe_align_block_size_pad",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
