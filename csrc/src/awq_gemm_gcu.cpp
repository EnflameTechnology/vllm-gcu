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

#include "awq_gemm_gcu.h"

#include <topsaten/topsaten_ops.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
// x, qweight, zeros, scale, zeros, split_k_iters, bias, group_size
at::Tensor awq_gemm_gcu(at::Tensor &input, at::Tensor &qweight,
                        at::Tensor &scale, at::Tensor &zeros,
                        int64_t split_k_iters,
                        const c10::optional<at::Tensor> &bias,
                        int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::Tensor out = at::empty({input.size(0), qweight.size(1)}, input.options());
  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }
  // int group_size = input.size(1) / scale.size(0);
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenGemmQuant)(
      out, input, qweight, scale, zeros, bias_tensor,
      static_cast<int>(group_size), stream));
  return out;
}
}  // namespace vllm_gcu::llm_ops
