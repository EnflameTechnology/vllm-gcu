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

#include "weak_ref_tensor.h"

#include <torch/all.h>

namespace vllm_gcu::llm_ops {

at::Tensor weak_ref_tensor(const at::Tensor& x) {
  if (x.numel() == 0) return x.alias();

  void* data_ptr = x.data_ptr();

  std::vector<int64_t> sizes = x.sizes().vec();
  std::vector<int64_t> strides = x.strides().vec();

  auto options = x.options();

  auto new_x = torch::from_blob(data_ptr, sizes, strides, options);

  return new_x;
}

}  // namespace vllm_gcu::llm_ops
