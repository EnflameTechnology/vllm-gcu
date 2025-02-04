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

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void reshape_and_cache(const at::Tensor &key, const at::Tensor &value,
                       at::Tensor &key_cache, at::Tensor &value_cache,
                       const at::Tensor &slot_mapping,
                       const std::string &kv_cache_dtype, double k_scale,
                       double v_scale, double k_zero, double v_zero) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(key));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  const char *kv_dtype = kv_cache_dtype.c_str();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmReshapeAndCacheQuant)(
      key_cache, value_cache, key, value, slot_mapping, kv_dtype, k_scale,
      k_zero, v_scale, v_zero, stream));
}

}  // namespace vllm_gcu::llm_ops
