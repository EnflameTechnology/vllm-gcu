
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

#include "rms_norm_static_fp8_quant.h"

#include <torch/all.h>

namespace vllm_gcu::llm_ops {

void rms_norm_static_fp8_quant(at::Tensor& result, const at::Tensor& input,
                               const at::Tensor& weight,
                               const at::Tensor& scale, double epsilon) {
  TORCH_CHECK(false, "rms_norm_static_fp8_quant is not supported yet.")
}

}  // namespace vllm_gcu::llm_ops
