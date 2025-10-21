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

#include "silu_mul_fp8_quant_deep_gemm.h"

#include <topsaten/topsaten_vllm.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void silu_mul_fp8_quant_deep_gemm(const at::Tensor& input,
                                  const at::Tensor& counts, at::Tensor& y_q,
                                  at::Tensor& y_s, int64_t group_size,
                                  bool use_ue8m0, int64_t num_parallel_tokens) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmSiluMulFp8QuantDeepGemm)(
      y_q, y_s, input, counts, group_size, use_ue8m0, stream));
}

}  // namespace vllm_gcu::llm_ops
