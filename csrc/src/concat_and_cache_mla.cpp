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

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void concat_and_cache_mla(const at::Tensor &kv_c, const at::Tensor &k_pe,
                          at::Tensor &kv_cache, const at::Tensor &slot_mapping,
                          c10::string_view kv_cache_dtype,
                          const at::Tensor &scale) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(kv_c));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor scale_tensor = scale;
  if (scale.dim() == 0) {
    scale_tensor = scale.unsqueeze(0);
  }

  const char *kv_dtype = kv_cache_dtype.data();
  //   const float scale_value = 1.0f;
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmConcatAndCacheMla)(
      kv_cache, kv_c, k_pe, slot_mapping, kv_dtype, scale_tensor, stream));
}

}  // namespace vllm_gcu::llm_ops
