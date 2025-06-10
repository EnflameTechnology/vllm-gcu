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

#include "dynamic_split.h"

#include <topsaten/topsaten_extensions.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void dynamic_split_gcu(at::TensorList out, const at::Tensor& input,
                       const at::Tensor& size, at::IntArrayRef split_sizes,
                       int64_t dim) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  topsatenSize_t split_sizes_t = {split_sizes.data(),
                                  static_cast<int64_t>(split_sizes.size())};

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsDynamicSplit)(
      out, input, size, split_sizes_t, dim, stream));
}

void dynamic_split(at::TensorList out, const at::Tensor& input,
                   const at::Tensor& size, at::IntArrayRef split_sizes,
                   int64_t dim) {
#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  std::vector<at::Tensor> out_cpu;
  at::Tensor input_cpu, size_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("dynamic_split") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("dynamic_split",
                            "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      out_cpu.reserve(out.size());
      for (const auto& tensor : out) {
        out_cpu.push_back(tensor.to(at::kCPU));
      }
      input_cpu = input.to(at::kCPU);
      size_cpu = size.to(at::kCPU);

      // Convert split_sizes to vector
      std::vector<int64_t> split_sizes_vec(split_sizes.begin(),
                                           split_sizes.end());

      // Call native implementation on CPU tensors
      extsDynamicSplit(out_cpu, input_cpu, size_cpu, split_sizes_vec, dim);

      VLLM_FALLBACK_CPU_LOG("dynamic_split",
                            "CPU fallback computation completed");
    }
  }
#endif

  dynamic_split_gcu(out, input, size, split_sizes, dim);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("dynamic_split", "Starting result verification");

    auto concat_out_cpu = torch::cat(out_cpu, 0);
    auto concat_out = torch::cat(out, 0);

    auto cpu_output = std::make_tuple(concat_out_cpu);
    auto device_outputs = std::make_tuple(concat_out.to(at::kCPU));
    EXPECT_TRUE(extsDynamicSplitCheck(cpu_output, device_outputs),
                "dynamic_split");

    // Copy results back to original tensors
    for (size_t i = 0; i < out.size(); ++i) {
      out[i].copy_(out_cpu[i]);
    }

    VLLM_FALLBACK_CPU_LOG("dynamic_split",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
