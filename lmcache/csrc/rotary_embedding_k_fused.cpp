/*
 * Copyright 2022-2025 Enflame. All Rights Reserved.

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

#include <topslmc/topslmc.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"  // NOLINT(build/include_subdir)

namespace lmcache {

void rotary_embedding_k_fused(const at::Tensor& old_positions,
                              const at::Tensor& new_positions, at::Tensor& key,
                              int64_t head_size,
                              const at::Tensor& cos_sin_cache, bool is_noex) {
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topslmc::topslmcRotaryEmbeddingKFused)(
      key, old_positions, new_positions, cos_sin_cache,
      static_cast<int>(head_size), is_noex, stream));
}

}  // namespace lmcache
