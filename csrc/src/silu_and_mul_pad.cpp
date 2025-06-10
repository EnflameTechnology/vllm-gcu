/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

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

#include <topsaten/topsaten_extensions.h>

#include <tuple>

#include "silu_and_mul.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void silu_and_mul_pad_gcu(at::Tensor &out, const at::Tensor &input,
                          const at::Tensor &size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsexts::topsextSiluAndMul)(out, input, size, stream));
}

void silu_and_mul_pad(at::Tensor &out, const at::Tensor &input,
                      const at::Tensor &size) {
  auto use_native = c10::utils::check_env("VLLM_GCU_NATIVE");
  auto fallback_cpu = c10::utils::check_env("VLLM_GCU_FALLBACK_CPU");

  if (use_native) {
    return;
  }

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor out_cpu, input_cpu, size_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("silu_and_mul_pad") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("silu_and_mul_pad",
                            "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      out_cpu = out.to(at::kCPU);
      input_cpu = input.to(at::kCPU);
      size_cpu = size.to(at::kCPU);

      // Call native implementation on CPU tensors
      extSiluAndMul(out_cpu, input_cpu, size_cpu);

      VLLM_FALLBACK_CPU_LOG("silu_and_mul_pad",
                            "CPU fallback computation completed");
    }
  }
#endif

  silu_and_mul_pad_gcu(out, input, size);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("silu_and_mul_pad", "Starting result verification");
    auto cpu_output = std::make_tuple(out_cpu);
    auto device_outputs = std::make_tuple(out.to(at::kCPU));
    EXPECT_TRUE(extSiluAndMulCheck(cpu_output, device_outputs),
                "silu_and_mul_pad");
    out.copy_(out_cpu);
    VLLM_FALLBACK_CPU_LOG("silu_and_mul_pad",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
