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
#include "concat_and_cache_mla.h"

#include <topsaten/topsaten_vllm.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void concat_and_cache_mla_gcu(const at::Tensor &kv_c, const at::Tensor &k_pe,
                              at::Tensor &kv_cache,
                              const at::Tensor &slot_mapping,
                              const char *kv_dtype,
                              const at::Tensor &scale_tensor) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(kv_c));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmConcatAndCacheMla)(
      kv_cache, kv_c, k_pe, slot_mapping, kv_dtype, scale_tensor, stream));
}

void concat_and_cache_mla(const at::Tensor &kv_c, const at::Tensor &k_pe,
                          at::Tensor &kv_cache, const at::Tensor &slot_mapping,
                          c10::string_view kv_cache_dtype,
                          const at::Tensor &scale) {
  at::Tensor scale_tensor = scale;
  if (scale.dim() == 0) {
    scale_tensor = scale.unsqueeze(0);
  }

  const char *kv_dtype = kv_cache_dtype.data();

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor kv_cache_cpu, kv_c_cpu, k_pe_cpu, slot_mapping_cpu,
      scale_tensor_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("concat_and_cache_mla") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Convert tensors to CPU for native implementation
      kv_cache_cpu = kv_cache.to(at::kCPU);
      kv_c_cpu = kv_c.to(at::kCPU);
      k_pe_cpu = k_pe.to(at::kCPU);
      slot_mapping_cpu = slot_mapping.to(at::kCPU);
      scale_tensor_cpu = scale_tensor.to(at::kCPU);

      // Call native implementation on CPU tensors
      vllmConcatAndCacheMla(kv_cache_cpu, kv_c_cpu, k_pe_cpu, slot_mapping_cpu,
                            kv_dtype, scale_tensor_cpu);
    }
  }
#endif

  concat_and_cache_mla_gcu(kv_c, k_pe, kv_cache, slot_mapping, kv_dtype,
                           scale_tensor);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    auto cpu_output = std::make_tuple(kv_cache_cpu);
    auto device_outputs = std::make_tuple(kv_cache.to(at::kCPU));
    EXPECT_TRUE(vllmConcatAndCacheMlaCheck(cpu_output, device_outputs),
                "concat_and_cache_mla");
    kv_cache.copy_(kv_cache_cpu);
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
