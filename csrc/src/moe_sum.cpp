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

#include "moe_sum.h"

#include <topsaten/topsaten_extensions.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
topsatenDataType_t scalarTypeToTopsatenDataType(
    const c10::ScalarType &scalar_type) {
  // auto narrow_dtype = get_gcu_scalar_type(scalar_type);
  auto narrow_dtype = scalar_type;
  switch (narrow_dtype) {
    // case c10::ScalarType::Bool:
    //   return TOPSATEN_DATA_PRED;
    // case c10::ScalarType::Byte:
    //   return TOPSATEN_DATA_U8;
    // case c10::ScalarType::UInt16:
    //   return TOPSATEN_DATA_U16;
    // case c10::ScalarType::UInt32:
    //   return TOPSATEN_DATA_U32;
    // case c10::ScalarType::Char:
    //   return TOPSATEN_DATA_I8;
    // case c10::ScalarType::Short:
    //   return TOPSATEN_DATA_I16;
    // case c10::ScalarType::Int:
    //   return TOPSATEN_DATA_I32;
    case c10::ScalarType::Half:
      return TOPSATEN_DATA_FP16;
    case c10::ScalarType::BFloat16:
      return TOPSATEN_DATA_BF16;
    case c10::ScalarType::Float:
      return TOPSATEN_DATA_FP32;
    // case c10::ScalarType::ComplexHalf:
    //   return TOPSATEN_DATA_CFP16;
    // case c10::ScalarType::ComplexFloat:
    //   return TOPSATEN_DATA_CFP32;
    default: {
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", narrow_dtype,
                            " to topsatenDataType_t.");
      return TOPSATEN_DATA_FP32;
    }
  }
}

void moe_sum_gcu(at::Tensor &out, const at::Tensor &input,
                 const at::Tensor &size, int64_t dim, bool keepdim) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::ScalarType dtype_aten = out.scalar_type();
  topsatenDataType_t dtype = scalarTypeToTopsatenDataType(dtype_aten);
  topsatenSize_t reduce_dim;
  std::vector<int64_t> reduce_dim_v = {dim};
  reduce_dim.data = reduce_dim_v.data();
  reduce_dim.len = reduce_dim_v.size();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsexts::topsextsSum)(
      out, input, size, reduce_dim, keepdim, dtype, stream));
}

void moe_sum(at::Tensor &out, const at::Tensor &input, const at::Tensor &size,
             int64_t dim, bool keepdim) {
#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor out_cpu, input_cpu, size_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("moe_sum") != std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("moe_sum", "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      out_cpu = out.to(at::kCPU);
      input_cpu = input.to(at::kCPU);
      size_cpu = size.to(at::kCPU);

      // Convert reduce_dim to IntArrayRef for native call
      std::vector<int64_t> reduce_dim_v = {dim};
      at::IntArrayRef dimensions(reduce_dim_v);
      at::ScalarType dtype_aten = out.scalar_type();

      // Call native implementation on CPU tensors
      extsSum(out_cpu, input_cpu, size_cpu, dimensions, keepdim, dtype_aten,
              nullptr);

      VLLM_FALLBACK_CPU_LOG("moe_sum", "CPU fallback computation completed");
    }
  }
#endif

  moe_sum_gcu(out, input, size, dim, keepdim);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("moe_sum", "Starting result verification");

    auto cpu_output = std::make_tuple(out_cpu);
    auto device_outputs = std::make_tuple(out.to(at::kCPU));
    EXPECT_TRUE(extsSumCheck(cpu_output, device_outputs), "moe_sum");
    out.copy_(out_cpu);

    VLLM_FALLBACK_CPU_LOG("moe_sum",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops
