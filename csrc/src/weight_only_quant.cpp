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

#include "weight_only_quant.h"

#include <topsaten/topsaten_ops.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
// output, x, qweight, None, qscales
void weight_only_quant(at::Tensor &output, const at::Tensor &input,
                       const at::Tensor &qweight,
                       const c10::optional<at::Tensor> &bias,
                       const at::Tensor &scale, int64_t group_size = -1) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
  }
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenLinearQuant)(
      output, input, qweight, bias_tensor, scale, static_cast<int>(group_size),
      stream));
}

}  // namespace vllm_gcu::llm_ops
