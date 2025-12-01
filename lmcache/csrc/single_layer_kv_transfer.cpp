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

void single_layer_kv_transfer(at::Tensor& lmc_key_value_cache,
                              at::Tensor& vllm_key_value_cache,
                              at::Tensor& slot_mapping, bool direction,
                              bool token_major = false,
                              bool vllm_two_major = false) {
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topslmc::topslmcSingleLayerKVTransfer)(
      lmc_key_value_cache, vllm_key_value_cache[0], vllm_key_value_cache[1],
      slot_mapping, token_major, direction, stream));
}

}  // namespace lmcache
