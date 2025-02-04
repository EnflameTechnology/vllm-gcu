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

#include "batched_rotary_embedding.h"

#include <torch/all.h>

namespace vllm_gcu::llm_ops {

void batched_rotary_embedding(const at::Tensor& positions, at::Tensor& query,
                              at::Tensor& key, int64_t head_size,
                              const at::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              const at::Tensor& cos_sin_cache_offsets) {
  TORCH_CHECK(false, "batched_rotary_embedding is not supported yet.")
}

}  // namespace vllm_gcu::llm_ops
