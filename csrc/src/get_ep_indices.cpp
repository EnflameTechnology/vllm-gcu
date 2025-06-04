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

#include "get_ep_indices.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void get_ep_indices_gcu(at::Tensor &ep_count, at::Tensor &ep_token_indices,
                        at::Tensor &ep_valid_token_indices,
                        const at::Tensor &topk_ids, int64_t expert_per_rank,
                        int64_t ep_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(topk_ids));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGetEPIndices)(
      ep_count, ep_token_indices, ep_valid_token_indices, topk_ids,
      static_cast<int>(expert_per_rank), static_cast<int>(ep_size), stream));
}
// x, qweight, zeros, scale, g_idx, bit, bias, group_size
void get_ep_indices(at::Tensor &ep_count, at::Tensor &ep_token_indices,
                    at::Tensor &ep_valid_token_indices,
                    const at::Tensor &topk_ids, int64_t expert_per_rank,
                    int64_t ep_size) {
#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor ep_count_cpu;
  at::Tensor ep_token_indices_cpu;
  at::Tensor ep_valid_token_indices_cpu;
  if (fallback_ops.has_value()) {
    if ((*fallback_ops).find("get_ep_indices") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      ep_count_cpu = ep_count.to(at::kCPU);
      ep_token_indices_cpu = ep_token_indices.to(at::kCPU);
      ep_valid_token_indices_cpu = ep_valid_token_indices.to(at::kCPU);
      at::Tensor topk_ids_cpu = topk_ids.to(at::kCPU);
      vllmGetEPIndices(ep_count_cpu, ep_token_indices_cpu,
                       ep_valid_token_indices_cpu, topk_ids_cpu,
                       static_cast<int>(expert_per_rank),
                       static_cast<int>(ep_size));
    }
  }
#endif

  get_ep_indices_gcu(ep_count, ep_token_indices, ep_valid_token_indices,
                     topk_ids, expert_per_rank, ep_size);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_outputs = std::make_tuple(ep_count_cpu, ep_token_indices_cpu,
                                       ep_valid_token_indices_cpu);
    auto device_outputs =
        std::make_tuple(ep_count.to(at::kCPU),
                        ep_token_indices.to(at::kCPU),
                        ep_valid_token_indices.to(at::kCPU));
    EXPECT_TRUE(vllmGetEPIndicesCheck(cpu_outputs, device_outputs),
                "get_ep_indices");
    ep_count.copy_(ep_count_cpu);
    ep_token_indices.copy_(ep_token_indices_cpu);
    ep_valid_token_indices.copy_(ep_valid_token_indices_cpu);
  }
#endif
}
}  // namespace vllm_gcu::llm_ops
