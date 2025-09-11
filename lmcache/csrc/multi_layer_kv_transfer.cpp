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

void multi_layer_kv_transfer(at::Tensor& key_value,
                             const at::Tensor& key_value_ptrs,
                             const at::Tensor& slot_mapping,
                             at::Device paged_memory_device,
                             int64_t page_buffer_size, bool direction,
                             bool use_mla) {
  const torch_gcu::OptionalGCUGuard device_guard(paged_memory_device);
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topslmc::topslmcMultiLayerKVTransfer)(
      key_value, key_value_ptrs, slot_mapping,
      static_cast<int>(page_buffer_size), direction, use_mla, stream));
}

}  // namespace lmcache
