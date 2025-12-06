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

#include "get_token_bin_counts_and_mask.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"


namespace vllm_gcu::llm_ops {
// bin_counts, mask, tokens, vocab_size, num_seqs
void get_token_bin_counts_and_mask(at::Tensor &bin_counts,
                                   at::Tensor &mask,
                                   const at::Tensor &tokens,
                                   const int64_t vocab_size,
                                   const int64_t num_seqs) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(tokens));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmGetTokenBinCountsAndMask)(
      bin_counts, mask, tokens, vocab_size, num_seqs, stream));
}
}  // namespace vllm_gcu::llm_ops
